use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use futures::channel::oneshot;
use futures_util::{stream, FutureExt, StreamExt};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use warp_multi_agent_api as api;

use super::super::{RequestParams, ResponseStream};
use crate::ai::agent::api::convert_conversation::convert_tool_call_result_to_input;
use crate::ai::agent::task::TaskId;
use crate::ai::agent::AIAgentInput;
use crate::ai::document::ai_document_model::{AIDocumentId, AIDocumentVersion};
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
    let client = reqwest::Client::builder()
        .danger_accept_invalid_certs(endpoint.allow_invalid_tls_certificates)
        .connect_timeout(Duration::from_secs(5))
        .build()?;
    let cancellation_rx = cancellation_rx.fuse();
    futures::pin_mut!(cancellation_rx);
    let mut attempt = 0;
    let response = loop {
        let mut builder = client.post(parsed_url.clone()).json(&request);
        if !endpoint.api_key.trim().is_empty() {
            builder = builder.bearer_auth(&endpoint.api_key);
        }
        let send = builder.send().fuse();
        futures::pin_mut!(send);
        let result = futures::select! {
            result = send => result,
            _ = cancellation_rx => anyhow::bail!("Local Agent request cancelled."),
        };
        match result {
            Ok(response) => break response,
            Err(error) if error.is_connect() && attempt < 4 => {
                let delay = tokio::time::sleep(Duration::from_secs(1 << attempt)).fuse();
                futures::pin_mut!(delay);
                futures::select! {
                    _ = delay => {},
                    _ = cancellation_rx => anyhow::bail!("Local Agent request cancelled."),
                }
                attempt += 1;
            }
            Err(error) => return Err(error.into()),
        }
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
        "content": "You are Warp's local terminal agent. Use tools to inspect the filesystem or run commands. Follow each tool's JSON schema exactly and include every required argument. Choose tools according to their documented semantics. Never claim a tool ran until its result is returned. If a tool returns an error, inspect it and retry with another appropriate tool or corrected arguments; do not replace the requested task with an unrelated response."
    })];
    for task in &params.tasks {
        let tool_calls = task
            .messages
            .iter()
            .filter_map(|message| match message.message.as_ref() {
                Some(api::message::Message::ToolCall(call)) => {
                    Some((call.tool_call_id.clone(), call))
                }
                _ => None,
            })
            .collect::<HashMap<_, _>>();
        let mut document_versions = HashMap::<AIDocumentId, AIDocumentVersion>::new();
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
                    let content = persisted_tool_result_content(
                        &task.id,
                        result,
                        &tool_calls,
                        &mut document_versions,
                    );
                    messages.push(serde_json::json!({
                        "role": "tool",
                        "tool_call_id": result.tool_call_id,
                        "content": content
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

fn persisted_tool_result_content(
    task_id: &str,
    result: &api::message::ToolCallResult,
    tool_calls: &HashMap<String, &api::message::ToolCall>,
    document_versions: &mut HashMap<AIDocumentId, AIDocumentVersion>,
) -> String {
    convert_tool_call_result_to_input(
        &TaskId::new(task_id.to_owned()),
        result,
        tool_calls,
        document_versions,
    )
    .and_then(|input| match input {
        AIAgentInput::ActionResult { result, .. } => Some(result.to_string()),
        _ => None,
    })
    .unwrap_or_else(|| "Tool result unavailable.".to_string())
}

fn tool_definitions() -> Vec<serde_json::Value> {
    vec![
        tool_definition(
            "read_files",
            "Read one or more local files",
            serde_json::json!({
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "paths": {
                        "type": "array",
                        "description": "File paths to read.",
                        "items": { "type": "string" },
                        "minItems": 1
                    }
                },
                "required": ["paths"]
            }),
        ),
        tool_definition(
            "file_glob",
            "Recursively search for files matching one glob pattern. This walks descendants and can be expensive in a large directory. Do not use it to list only the immediate entries of a directory; use run_shell_command with an appropriate directory-listing command instead.",
            serde_json::json!({
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Required filename glob pattern, for example *.rs. Matching is recursive below the search directory."
                    },
                    "path": {
                        "type": "string",
                        "description": "Optional directory to search. Omit to use the current working directory."
                    }
                },
                "required": ["pattern"]
            }),
        ),
        tool_definition(
            "grep",
            "Search file contents",
            serde_json::json!({
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "query": { "type": "string", "description": "Required text or regular expression to search for." },
                    "path": { "type": "string", "description": "Optional file or directory to search." }
                },
                "required": ["query"]
            }),
        ),
        tool_definition(
            "run_shell_command",
            "Run a shell command with Warp permission checks. Use this for shell-native operations such as listing the immediate entries in the current directory.",
            serde_json::json!({
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "command": { "type": "string", "description": "Required shell command to execute." }
                },
                "required": ["command"]
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
    let arguments = parse_tool_arguments(&call.function.arguments)?;
    let tool = match call.function.name.as_str() {
        "read_files" => {
            let paths = required_string_array(&arguments, "paths")?;
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
                patterns: vec![required_string(&arguments, "pattern")?],
                search_dir: optional_string(&arguments, "path"),
                ..Default::default()
            })
        }
        "grep" => api::message::tool_call::Tool::Grep(api::message::tool_call::Grep {
            queries: vec![required_string(&arguments, "query")?],
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

fn parse_tool_arguments(arguments: &str) -> anyhow::Result<serde_json::Value> {
    let parsed: serde_json::Value = serde_json::from_str(arguments)
        .map_err(|_| anyhow::anyhow!("The local model returned malformed tool arguments."))?;
    let parsed = match parsed {
        serde_json::Value::String(encoded) => serde_json::from_str(&encoded)
            .map_err(|_| anyhow::anyhow!("The local model returned malformed tool arguments."))?,
        parsed => parsed,
    };
    if !parsed.is_object() {
        anyhow::bail!("The local model returned malformed tool arguments.");
    }
    Ok(parsed)
}

fn required_string_array(value: &serde_json::Value, key: &str) -> anyhow::Result<Vec<String>> {
    value
        .get(key)
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("Missing or invalid tool argument `{key}`."))?
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
        .ok_or_else(|| anyhow::anyhow!("Missing or invalid tool argument `{key}`."))
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
            serde_json::json!({
                "pattern": tool.patterns.first().cloned().unwrap_or_default(),
                "path": tool.search_dir
            }),
        ),
        api::message::tool_call::Tool::Grep(tool) => (
            "grep",
            serde_json::json!({
                "query": tool.queries.first().cloned().unwrap_or_default(),
                "path": tool.path
            }),
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
        .local_task_id
        .clone()
        .or_else(|| params.tasks.first().map(|task| task.id.clone()))
        .unwrap_or_else(|| Uuid::new_v4().to_string());
    let message_types = params
        .input
        .iter()
        .filter_map(persisted_input_message)
        .chain(std::iter::once(message_type));
    let messages = message_types
        .map(|message_type| api::Message {
            id: Uuid::new_v4().to_string(),
            task_id: task_id.clone(),
            request_id: request_id.clone(),
            message: Some(message_type),
            ..Default::default()
        })
        .collect();
    let mut actions = Vec::new();
    if params.tasks.is_empty() {
        actions.push(api::ClientAction {
            action: Some(api::client_action::Action::CreateTask(
                api::client_action::CreateTask {
                    task: Some(api::Task {
                        id: task_id.clone(),
                        messages: vec![],
                        dependencies: None,
                        description: String::new(),
                        summary: String::new(),
                        server_data: String::new(),
                    }),
                },
            )),
        });
    }
    actions.push(api::ClientAction {
        action: Some(api::client_action::Action::AddMessagesToTask(
            api::client_action::AddMessagesToTask { task_id, messages },
        )),
    });
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
                api::response_event::ClientActions { actions },
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

#[allow(deprecated)]
fn persisted_input_message(input: &AIAgentInput) -> Option<api::message::Message> {
    use api::message::tool_call_result::Result as MessageResult;
    use api::request::input::tool_call_result::Result as RequestResult;
    use api::request::input::user_inputs::user_input::Input as RequestInput;

    let AIAgentInput::ActionResult { result, .. } = input else {
        return match input {
            AIAgentInput::UserQuery { query, .. } => {
                Some(api::message::Message::UserQuery(api::message::UserQuery {
                    query: query.clone(),
                    ..Default::default()
                }))
            }
            _ => None,
        };
    };
    let RequestInput::ToolCallResult(result) = result.clone().try_into().ok()? else {
        return None;
    };
    let result = match result.result? {
        RequestResult::RunShellCommand(result) => MessageResult::RunShellCommand(result),
        RequestResult::ReadFiles(result) => MessageResult::ReadFiles(result),
        RequestResult::Grep(result) => MessageResult::Grep(result),
        RequestResult::FileGlob(result) => MessageResult::FileGlob(result),
        RequestResult::FileGlobV2(result) => MessageResult::FileGlobV2(result),
        _ => return None,
    };

    Some(api::message::Message::ToolCallResult(
        api::message::ToolCallResult {
            tool_call_id: result_id(input)?.to_string(),
            context: None,
            result: Some(result),
        },
    ))
}

fn result_id(input: &AIAgentInput) -> Option<&crate::ai::agent::AIAgentActionId> {
    match input {
        AIAgentInput::ActionResult { result, .. } => Some(&result.id),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn response_events_target_the_request_input_task() {
        let mut params = RequestParams::new_for_test();
        params.local_task_id = Some("optimistic-root-task".to_string());

        let events = response_events(&params, "hello".to_string());
        let actions = match events[1].r#type.as_ref() {
            Some(api::response_event::Type::ClientActions(actions)) => actions,
            event => panic!("expected client actions, got {event:?}"),
        };
        assert!(matches!(
            actions.actions[0].action,
            Some(api::client_action::Action::CreateTask(_))
        ));
        let add = match actions.actions[1].action.as_ref() {
            Some(api::client_action::Action::AddMessagesToTask(add)) => add,
            action => panic!("expected AddMessagesToTask, got {action:?}"),
        };

        assert_eq!(add.task_id, "optimistic-root-task");
        assert_eq!(add.messages[0].task_id, "optimistic-root-task");
    }

    #[test]
    fn response_events_persist_action_results_before_the_next_agent_message() {
        let mut params = RequestParams::new_for_test();
        params.local_task_id = Some("root-task".to_string());
        params.input.push(AIAgentInput::ActionResult {
            result: crate::ai::agent::AIAgentActionResult {
                id: "tool-call-1".to_string().into(),
                task_id: TaskId::new("root-task".to_string()),
                result: crate::ai::agent::AIAgentActionResultType::FileGlobV2(
                    crate::ai::agent::FileGlobV2Result::Success {
                        matched_files: vec![crate::ai::agent::FileGlobV2Match {
                            file_path: "/tmp/example.rs".to_string(),
                        }],
                        warnings: None,
                    },
                ),
            },
            context: Arc::from([]),
        });

        let events = response_events(&params, "done".to_string());
        let actions = match events[1].r#type.as_ref() {
            Some(api::response_event::Type::ClientActions(actions)) => actions,
            event => panic!("expected client actions, got {event:?}"),
        };
        let add = match actions.actions[1].action.as_ref() {
            Some(api::client_action::Action::AddMessagesToTask(add)) => add,
            action => panic!("expected AddMessagesToTask, got {action:?}"),
        };

        assert!(matches!(
            add.messages[0].message,
            Some(api::message::Message::ToolCallResult(_))
        ));
        assert!(matches!(
            add.messages[1].message,
            Some(api::message::Message::AgentOutput(_))
        ));
    }

    #[test]
    fn response_events_persist_user_queries_before_the_agent_message() {
        let mut params = RequestParams::new_for_test();
        params.local_task_id = Some("root-task".to_string());
        params.input.push(AIAgentInput::UserQuery {
            query: "use the detected changes".to_string(),
            context: Arc::from([]),
            static_query_type: None,
            referenced_attachments: HashMap::new(),
            user_query_mode: Default::default(),
            running_command: None,
            intended_agent: None,
        });

        let events = response_events(&params, "creating the commit".to_string());
        let actions = match events[1].r#type.as_ref() {
            Some(api::response_event::Type::ClientActions(actions)) => actions,
            event => panic!("expected client actions, got {event:?}"),
        };
        let add = match actions.actions[1].action.as_ref() {
            Some(api::client_action::Action::AddMessagesToTask(add)) => add,
            action => panic!("expected AddMessagesToTask, got {action:?}"),
        };

        assert!(matches!(
            add.messages[0].message,
            Some(api::message::Message::UserQuery(ref query))
                if query.query == "use the detected changes"
        ));
        assert!(matches!(
            add.messages[1].message,
            Some(api::message::Message::AgentOutput(_))
        ));
    }

    #[test]
    fn tool_arguments_follow_the_advertised_canonical_schema() {
        assert_eq!(
            required_string(&serde_json::json!({ "pattern": "*.rs" }), "pattern").unwrap(),
            "*.rs"
        );
        assert_eq!(
            required_string_array(&serde_json::json!({ "paths": ["a.rs", "b.toml"] }), "paths")
                .unwrap(),
            vec!["a.rs", "b.toml"]
        );
        assert!(required_string(&serde_json::json!({}), "pattern").is_err());
        assert!(required_string(&serde_json::json!({ "cmd": "pwd" }), "command").is_err());
    }

    #[test]
    fn tool_arguments_accept_openai_json_and_double_encoded_json() {
        let direct = parse_tool_arguments(r#"{"pattern":"*"}"#).unwrap();
        let encoded = parse_tool_arguments(r#""{\"pattern\":\"*\"}""#).unwrap();

        assert_eq!(direct, serde_json::json!({ "pattern": "*" }));
        assert_eq!(encoded, direct);
    }

    #[test]
    fn tool_descriptions_distinguish_recursive_search_from_directory_listing() {
        let tools = tool_definitions();
        let description = |name: &str| {
            tools
                .iter()
                .find(|tool| tool["function"]["name"] == name)
                .and_then(|tool| tool["function"]["description"].as_str())
                .unwrap()
        };

        assert!(description("file_glob").contains("Recursively search"));
        assert!(description("file_glob").contains("Do not use it to list"));
        assert!(description("run_shell_command").contains("listing the immediate entries"));
    }
}
