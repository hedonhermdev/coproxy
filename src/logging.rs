use serde_json::Value;

const REDACTED_KEYS: &[&str] = &[
    "authorization",
    "api_key",
    "apikey",
    "token",
    "access_token",
    "github_token",
    "ghcp_token",
    "x-api-key",
];

pub fn redact_json(value: &Value) -> String {
    let mut cloned = value.clone();
    redact_in_place(&mut cloned);
    cloned.to_string()
}

pub fn redact_serializable<T: serde::Serialize>(req: &T) -> String {
    match serde_json::to_value(req) {
        Ok(v) => redact_json(&v),
        Err(e) => format!("<serialize error: {e}>"),
    }
}

fn redact_in_place(v: &mut Value) {
    match v {
        Value::Object(map) => {
            for (k, val) in map.iter_mut() {
                if REDACTED_KEYS.iter().any(|r| k.eq_ignore_ascii_case(r)) {
                    *val = Value::String("[REDACTED]".into());
                } else {
                    redact_in_place(val);
                }
            }
        }
        Value::Array(arr) => arr.iter_mut().for_each(redact_in_place),
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn redacts_known_keys_case_insensitively() {
        let v = json!({
            "model": "gpt-4o",
            "api_key": "sk-leak",
            "Authorization": "Bearer xyz",
            "messages": [{"role": "user", "content": "hi"}],
        });
        let out = redact_json(&v);
        assert!(out.contains("\"[REDACTED]\""));
        assert!(!out.contains("sk-leak"));
        assert!(!out.contains("Bearer xyz"));
        assert!(out.contains("gpt-4o"));
    }

    #[test]
    fn redacts_nested_keys() {
        let v = json!({
            "outer": {"token": "abc", "inner": {"github_token": "ghp_x"}},
        });
        let out = redact_json(&v);
        assert!(!out.contains("abc"));
        assert!(!out.contains("ghp_x"));
    }
}
