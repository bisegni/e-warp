use super::*;

const REMOTE: CustomEndpointReachability = CustomEndpointReachability::RemoteServerReachable;
const LOCAL: CustomEndpointReachability = CustomEndpointReachability::LocalClientReachable;

#[test]
fn validate_url_accepts_https_with_host() {
    assert!(validate_url("https://api.example.com/v1", REMOTE).is_ok());
    assert!(validate_url("https://example.com", REMOTE).is_ok());
    assert!(validate_url("https://8.8.8.8/v1", REMOTE).is_ok());
}

#[test]
fn validate_url_rejects_http() {
    assert_eq!(
        validate_url("http://api.example.com/v1", REMOTE),
        Err("URL must use HTTPS")
    );
    assert_eq!(
        validate_url("http://example.com", REMOTE),
        Err("URL must use HTTPS")
    );
}

#[test]
fn validate_url_rejects_ftp_and_other_schemes() {
    assert_eq!(
        validate_url("ftp://files.example.com", REMOTE),
        Err("URL must use HTTPS")
    );
    assert_eq!(
        validate_url("file:///etc/passwd", REMOTE),
        Err("URL must use HTTPS")
    );
    assert_eq!(
        validate_url("ws://socket.example.com", REMOTE),
        Err("URL must use HTTPS")
    );
}

#[test]
fn validate_url_rejects_malformed_strings() {
    assert_eq!(validate_url("not a url", REMOTE), Err("Invalid URL"));
    assert_eq!(validate_url("https://", REMOTE), Err("Invalid URL"));
}

#[test]
fn validate_url_rejects_empty_host() {
    assert_eq!(validate_url("https://?query=1", REMOTE), Err("Invalid URL"));
}

#[test]
fn validate_url_allows_empty_string() {
    assert!(validate_url("", REMOTE).is_ok());
}

#[test]
fn validate_url_allows_whitespace_only() {
    assert!(validate_url("   ", REMOTE).is_ok());
}

#[test]
fn validate_url_rejects_localhost_and_private_ips() {
    let error = Err("URL must not use a local or private host");
    assert_eq!(validate_url("https://localhost:8080", REMOTE), error);
    assert_eq!(validate_url("https://127.0.0.1/v1", REMOTE), error);
    assert_eq!(validate_url("https://0.0.0.0/v1", REMOTE), error);
    assert_eq!(validate_url("https://10.0.0.1/v1", REMOTE), error);
    assert_eq!(validate_url("https://172.16.0.1/v1", REMOTE), error);
    assert_eq!(validate_url("https://192.168.0.1/v1", REMOTE), error);
    assert_eq!(validate_url("https://169.254.0.1/v1", REMOTE), error);
    assert_eq!(validate_url("https://[::1]/v1", REMOTE), error);
    assert_eq!(validate_url("https://[::]/v1", REMOTE), error);
    assert_eq!(validate_url("https://[fc00::1]/v1", REMOTE), error);
    assert_eq!(validate_url("https://[fe80::1]/v1", REMOTE), error);
    assert_eq!(
        validate_url("https://[::ffff:192.168.0.1]/v1", REMOTE),
        error
    );
}

#[test]
fn validate_url_local_mode_accepts_local_and_private_hosts() {
    assert!(validate_url("https://slacstudio.local:4443/v1", LOCAL).is_ok());
    assert!(validate_url("http://localhost:4000/v1", LOCAL).is_ok());
    assert!(validate_url("http://127.0.0.1/v1", LOCAL).is_ok());
    assert!(validate_url("https://10.0.0.1/v1", LOCAL).is_ok());
    assert!(validate_url("https://192.168.1.12/v1", LOCAL).is_ok());
    assert!(validate_url("http://[::1]/v1", LOCAL).is_ok());
}

#[test]
fn validate_url_local_mode_rejects_non_http_schemes() {
    assert_eq!(
        validate_url("ftp://localhost/v1", LOCAL),
        Err("URL must use HTTP or HTTPS")
    );
}

#[test]
fn endpoint_form_valid_rejects_invalid_current_url() {
    assert!(!is_endpoint_form_valid(
        "Endpoint",
        "http://api.example.com/v1",
        "key",
        true
    ));
}

#[test]
fn endpoint_form_valid_requires_non_empty_url() {
    assert!(!is_endpoint_form_valid("Endpoint", "", "key", true));
    assert!(!is_endpoint_form_valid("Endpoint", "   ", "key", true));
}

#[test]
fn endpoint_form_valid_accepts_complete_valid_form() {
    assert!(is_endpoint_form_valid(
        "Endpoint",
        "https://api.example.com/v1",
        "key",
        true
    ));
}

#[test]
fn endpoint_form_valid_accepts_local_mode_local_url() {
    assert!(is_endpoint_form_valid_for_reachability(
        "Endpoint",
        "http://localhost:4000/v1",
        "key",
        true,
        LOCAL
    ));
}

#[test]
fn endpoint_form_valid_accepts_local_mode_without_api_key() {
    assert!(is_endpoint_form_valid_for_reachability(
        "Ollama",
        "http://localhost:11434/v1",
        "",
        true,
        LOCAL
    ));
}

#[test]
fn endpoint_form_valid_requires_api_key_for_remote_mode() {
    assert!(!is_endpoint_form_valid_for_reachability(
        "Remote",
        "https://api.example.com/v1",
        "",
        true,
        REMOTE
    ));
}
