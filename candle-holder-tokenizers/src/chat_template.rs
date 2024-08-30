use candle_holder::{Error, Result};
use minijinja::{Environment, ErrorKind, Template};
use minijinja_contrib::pycompat;
use serde::{Deserialize, Serialize};

/// Represents a message in a conversation between a user and an assistant.
#[derive(Debug, Serialize, Deserialize)]
pub struct Message {
    /// The role of the message. Can be "system", "user", or "assistant".
    role: String,
    /// The content of the message.
    content: String,
}

impl Message {
    /// Creates a system prompt message.
    pub fn system<S: Into<String>>(content: S) -> Self {
        Self {
            role: "system".to_string(),
            content: content.into(),
        }
    }

    /// Creates a user message.
    pub fn user<S: Into<String>>(content: S) -> Self {
        Self {
            role: "user".to_string(),
            content: content.into(),
        }
    }

    /// Creates an assistant message.
    pub fn assistant<S: Into<String>>(content: S) -> Self {
        Self {
            role: "assistant".to_string(),
            content: content.into(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct ChatTemplateInputs {
    messages: Vec<Message>,
    bos_token: Option<String>,
    eos_token: Option<String>,
}

/// https://github.com/huggingface/text-generation-inference/blob/d9fbbaafb046bb423e31edaf9ccf8eecc2d5c33d/router/src/infer/chat_template.rs#L4
#[derive(Debug)]
pub struct ChatTemplate {
    template: Template<'static, 'static>,
    bos_token: Option<String>,
    eos_token: Option<String>,
}

impl ChatTemplate {
    pub fn new(
        template: String,
        bos_token: Option<String>,
        eos_token: Option<String>,
    ) -> Result<Self> {
        let mut env = Box::new(Environment::new());
        env.set_unknown_method_callback(pycompat::unknown_method_callback);
        env.add_function("raise_exception", raise_exception);

        // Using `Box::leak` and `'static` lifetime implies that the template will live for the
        // duration of the program.
        let template = Box::leak(env)
            .template_from_str(Box::leak(template.into_boxed_str()))
            .unwrap();

        Ok(Self {
            template,
            bos_token,
            eos_token,
        })
    }

    pub fn apply(&self, messages: Vec<Message>) -> Result<String> {
        self.template
            .render(&ChatTemplateInputs {
                messages,
                bos_token: self.bos_token.clone(),
                eos_token: self.eos_token.clone(),
            })
            .map_err(Error::wrap)
    }
}

fn raise_exception(err_text: String) -> std::result::Result<String, minijinja::Error> {
    Err(minijinja::Error::new(ErrorKind::SyntaxError, err_text))
}

#[cfg(test)]
mod tests {
    use super::{ChatTemplate, Message};

    #[test]
    fn test_chat_template() {
        let template = "{% for message in messages %}{{ message.content }} {{ message.role }}: {{ message.content }}\n{% endfor %}".to_string();
        let chat_template = ChatTemplate::new(template, None, None).unwrap();
        let messages = vec![
            Message {
                role: "user".to_string(),
                content: "Hello".to_string(),
            },
            Message {
                role: "assistant".to_string(),
                content: "Hi".to_string(),
            },
        ];
        let result = chat_template.apply(messages).unwrap();
        assert_eq!("Hello user: Hello\nHi assistant: Hi\n", result);
    }
}
