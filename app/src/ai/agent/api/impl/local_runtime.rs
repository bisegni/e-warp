use std::sync::Arc;

use futures::channel::oneshot;
use futures_util::{stream, StreamExt};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use warp_multi_agent_api as api;

use super::super::{RequestParams, ResponseStream};
use crate::ai::agent::AIAgentInput;
use crate::server::server_api::AIApiError;

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<serde_json::Value>,
    tools: Vec<serde_json::Value>,
    parallel_tool_calls: bool,
    stream: bool,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatResponseMessage,
}

#[derive(Deserialize)]
struct ChatResponseMessage {
    content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<OpenAIToolCall>,
}

#[derive(Deserialize)]
struct OpenAIToolCall {
    id: String,
    function: OpenAIFunctionCall,
}

#[derive(Deserialize)]
struct OpenAIFunctionCall {
    name: String,
    arguments: String,
}

pub(super) async fn generate(
    params: RequestParams,
    cancellation_rx: oneshot::Receiver<()>,
) -> ResponseStream {
    let output = stream::once(async move {
        match run(params, cancellation_rx).await {
            Ok(events) => events.into_iter().map(Ok).collect::<Vec<_>>(),
            Err(error) => vec![Err(Arc::new(AIApiError::Other(error)))],
        }
    })
    .flat_map(stream::iter);
    Box::pin(output)
}

async fn run(
    params: RequestParams,
    cancellation_rx: oneshot::Receiver<()>,
) -> anyhow::Result<Vec<api::ResponseEvent>> {
    let endpoint = params.local_custom_endpoint.as_ref().ok_or_else(|| {
        anyhow::anyhow!(
            "No usable local endpoint was resolved. Save an endpoint in local-device mode with at least one model."
        )
    })?;
    let config_key = params
        .local_custom_model_config_keys
        .first()
        .ok_or_else(|| anyhow::anyhow!("No local model was selected."))?;
    let model = endpoint
        .models
        .iter()
        .find(|model| &model.config_key == config_key)
        .ok_or_else(|| anyhow::anyhow!("The selected local model is no longer configured."))?;
    let query = params.input.iter().rev().find_map(|input| match input {
        AIAgentInput::UserQuery { query, .. } => Some(query.as_str()),
        _ => None,
    });

    let url = format!("{}/chat/completions", endpoint.url.trim_end_matches('/'));
    let parsed_url = reqwest::Url::parse(&url)
        .map_err(|_| anyhow::anyhow!("Local endpoint {} has an invalid URL.", endpoint.name))?;
    if http_client::classify_destination(&parsed_url)
        != http_client::DestinationClass::LocalEndpoint
    {
        anyhow::bail!(
            "Offline Agent Mode only permits localhost, private-network, and .local endpoints."
        );
    }
    let mut messages = conversation_messages(&params);
    if messages.len() == 1 {
        let query = query.ok_or_else(|| {
            anyhow::anyhow!("Local Agent Mode received no user prompt or tool result.")
        })?;
        messages.push(serde_json::json!({ "role": "user", "content": query }));
    }
    let request = ChatRequest {
        model: &model.name,
        messages,
        tools: tool_definitions(),
        parallel_tool_calls: false,
        stream: false,
    };
    let client = reqwest::Client::new();
    let mut builder = client.post(parsed_url).json(&request);
    if !endpoint.api_key.trim().is_empty() {
        builder = builder.bearer_auth(&endpoint.api_key);
    }
    let response = futures::future::select(
        Box::pin(async move { builder.send().await }),
        Box::pin(cancellation_rx),
    )
    .await;
    let response = match response {
        futures::future::Either::Left((response, _)) => response?,
        futures::future::Either::Right(_) => anyhow::bail!("Local Agent request cancelled."),
    };
    let status = response.status();
    let body = response.text().await?;
    if !status.is_success() {
        anyhow::bail!("Local endpoint {} returned HTTP {status}.", endpoint.name);
    }
    let response: ChatResponse = serde_json::from_str(&body).map_err(|_| {
        anyhow::anyhow!(
            "Local endpoint {} returned an invalid OpenAI-compatible response.",
            endpoint.name
        )
    })?;
    let choice = response.choices.into_iter().next().ok_or_else(|| {
        anyhow::anyhow!(
            "Local endpoint {} returned no completion choices.",
            endpoint.name
        )
    })?;
    if choice.message.tool_calls.len() > 1 {
        anyhow::bail!("Local Agent Mode supports one tool call at a time.");
    }
    if let Some(tool_call) = choice.message.tool_calls.into_iter().next() {
        return Ok(tool_call_events(&params, tool_call)?);
    }
    let text = choice
        .message
        .content
        .filter(|text| !text.is_empty())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "Local endpoint {} returned an empty response.",
                endpoint.name
            )
        })?;

    Ok(response_events(&params, text))
}

fn conversation_messages(params: &RequestParams) -> Vec<serde_json::Value> {
    let mut messages = vec![serde_json::json!({
        "role": "system",
        "content": "You are Warp's local terminal agent. Use tools to inspect the filesystem or run commands. Never claim a tool ran until its result is returned."
    })];
    for task in &params.tasks {
        for message in &task.messages {
            match message.message.as_ref() {
                Some(api::message::Message::UserQuery(query)) => {
                    messages.push(serde_json::json!({ "role": "user", "content": query.query }))
                }
                Some(api::message::Message::AgentOutput(output)) => messages
                    .push(serde_json::json!({ "role": "assistant", "content": output.text })),
                Some(api::message::Message::ToolCall(call)) => {
                    if let Some((name, arguments)) = tool_call_to_openai(call) {
                        messages.push(serde_json::json!({
                            "role": "assistant",
                            "content": null,
                            "tool_calls": [{
                                "id": call.tool_call_id,
                                "type": "function",
                                "function": { "name": name, "arguments": arguments }
                            }]
                        }));
                    }
                }
                Some(api::message::Message::ToolCallResult(result)) => {
                    messages.push(serde_json::json!({
                        "role": "tool",
                        "tool_call_id": result.tool_call_id,
                        "content": format!("{result:?}")
                    }))
                }
                _ => {}
            }
        }
    }
    for input in &params.input {
        match input {
            AIAgentInput::UserQuery { query, .. } => {
                messages.push(serde_json::json!({ "role": "user", "content": query }));
            }
            AIAgentInput::ActionResult { result, .. } => {
                messages.push(serde_json::json!({
                    "role": "tool",
                    "tool_call_id": result.id.to_string(),
                    "content": result.to_string()
                }));
            }
            _ => {}
        }
    }
    messages
}

fn tool_definitions() -> Vec<serde_json::Value> {
    vec![
        tool_definition(
            "read_files",
            "Read one or more local files",
            serde_json::json!({
                "type": "object", "properties": { "paths": { "type": "array", "items": { "type": "string" } } }, "required": ["paths"]
            }),
        ),
        tool_definition(
            "file_glob",
            "Find files by glob patterns",
            serde_json::json!({
                "type": "object", "properties": { "patterns": { "type": "array", "items": { "type": "string" } }, "path": { "type": "string" } }, "required": ["patterns"]
            }),
        ),
        tool_definition(
            "grep",
            "Search file contents",
            serde_json::json!({
                "type": "object", "properties": { "queries": { "type": "array", "items": { "type": "string" } }, "path": { "type": "string" } }, "required": ["queries"]
            }),
        ),
        tool_definition(
            "run_shell_command",
            "Run a shell command with Warp permission checks",
            serde_json::json!({
                "type": "object", "properties": { "command": { "type": "string" } }, "required": ["command"]
            }),
        ),
    ]
}

fn tool_definition(
    name: &str,
    description: &str,
    parameters: serde_json::Value,
) -> serde_json::Value {
    serde_json::json!({
        "type": "function",
        "function": { "name": name, "description": description, "parameters": parameters }
    })
}

fn tool_call_events(
    params: &RequestParams,
    call: OpenAIToolCall,
) -> anyhow::Result<Vec<api::ResponseEvent>> {
    let arguments: serde_json::Value = serde_json::from_str(&call.function.arguments)
        .map_err(|_| anyhow::anyhow!("The local model returned malformed tool arguments."))?;
    let tool = match call.function.name.as_str() {
        "read_files" => {
            let paths = string_array(&arguments, "paths")?;
            api::message::tool_call::Tool::ReadFiles(api::message::tool_call::ReadFiles {
                files: paths
                    .into_iter()
                    .map(|name| api::message::tool_call::read_files::File {
                        name,
                        line_ranges: vec![],
                    })
                    .collect(),
            })
        }
        "file_glob" => {
            api::message::tool_call::Tool::FileGlobV2(api::message::tool_call::FileGlobV2 {
                patterns: string_array(&arguments, "patterns")?,
                search_dir: optional_string(&arguments, "path"),
                ..Default::default()
            })
        }
        "grep" => api::message::tool_call::Tool::Grep(api::message::tool_call::Grep {
            queries: string_array(&arguments, "queries")?,
            path: optional_string(&arguments, "path"),
        }),
        "run_shell_command" => api::message::tool_call::Tool::RunShellCommand(
            api::message::tool_call::RunShellCommand {
                command: required_string(&arguments, "command")?,
                ..Default::default()
            },
        ),
        _ => anyhow::bail!("The local model requested an unsupported tool."),
    };
    Ok(message_events(
        params,
        api::message::Message::ToolCall(api::message::ToolCall {
            tool_call_id: call.id,
            tool: Some(tool),
        }),
    ))
}

fn string_array(value: &serde_json::Value, key: &str) -> anyhow::Result<Vec<String>> {
    value
        .get(key)
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("Missing tool argument `{key}`."))?
        .iter()
        .map(|item| {
            item.as_str()
                .map(str::to_owned)
                .ok_or_else(|| anyhow::anyhow!("Invalid tool argument `{key}`."))
        })
        .collect()
}

fn required_string(value: &serde_json::Value, key: &str) -> anyhow::Result<String> {
    value
        .get(key)
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| anyhow::anyhow!("Missing tool argument `{key}`."))
}

fn optional_string(value: &serde_json::Value, key: &str) -> String {
    value
        .get(key)
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .to_owned()
}

fn tool_call_to_openai(call: &api::message::ToolCall) -> Option<(&'static str, String)> {
    let (name, arguments) = match call.tool.as_ref()? {
        api::message::tool_call::Tool::ReadFiles(tool) => (
            "read_files",
            serde_json::json!({ "paths": tool.files.iter().map(|file| &file.name).collect::<Vec<_>>() }),
        ),
        api::message::tool_call::Tool::FileGlobV2(tool) => (
            "file_glob",
            serde_json::json!({ "patterns": tool.patterns, "path": tool.search_dir }),
        ),
        api::message::tool_call::Tool::Grep(tool) => (
            "grep",
            serde_json::json!({ "queries": tool.queries, "path": tool.path }),
        ),
        api::message::tool_call::Tool::RunShellCommand(tool) => (
            "run_shell_command",
            serde_json::json!({ "command": tool.command }),
        ),
        _ => return None,
    };
    Some((name, arguments.to_string()))
}

fn response_events(params: &RequestParams, text: String) -> Vec<api::ResponseEvent> {
    message_events(
        params,
        api::message::Message::AgentOutput(api::message::AgentOutput { text }),
    )
}

fn message_events(
    params: &RequestParams,
    message_type: api::message::Message,
) -> Vec<api::ResponseEvent> {
    let request_id = Uuid::new_v4().to_string();
    let conversation_id = params
        .conversation_token
        .as_ref()
        .map(|token| token.as_str().to_owned())
        .unwrap_or_else(|| Uuid::new_v4().to_string());
    let task_id = params
        .tasks
        .first()
        .map(|task| task.id.clone())
        .unwrap_or_else(|| Uuid::new_v4().to_string());
    let message = api::Message {
        id: Uuid::new_v4().to_string(),
        task_id: task_id.clone(),
        request_id: request_id.clone(),
        message: Some(message_type),
        ..Default::default()
    };
    vec![
        api::ResponseEvent {
            r#type: Some(api::response_event::Type::Init(
                api::response_event::StreamInit {
                    conversation_id,
                    request_id: request_id.clone(),
                    run_id: String::new(),
                },
            )),
        },
        api::ResponseEvent {
            r#type: Some(api::response_event::Type::ClientActions(
                api::response_event::ClientActions {
                    actions: vec![api::ClientAction {
                        action: Some(api::client_action::Action::AddMessagesToTask(
                            api::client_action::AddMessagesToTask {
                                task_id,
                                messages: vec![message],
                            },
                        )),
                    }],
                },
            )),
        },
        api::ResponseEvent {
            r#type: Some(api::response_event::Type::Finished(
                api::response_event::StreamFinished {
                    reason: Some(api::response_event::stream_finished::Reason::Done(
                        api::response_event::stream_finished::Done {},
                    )),
                    ..Default::default()
                },
            )),
        },
    ]
}
