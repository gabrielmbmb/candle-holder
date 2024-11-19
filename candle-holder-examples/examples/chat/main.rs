#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use std::{
    io::{self, Write},
    sync::{mpsc, Arc, Mutex},
};

use anyhow::{Error, Result};
use candle_holder_examples::Cli;
use candle_holder_models::{
    AutoModelForCausalLM, GenerationConfig, GenerationParams, TextIteratorStreamer, TokenStreamer,
};
use candle_holder_tokenizers::{AutoTokenizer, Message};
use clap::Parser;

#[derive(Debug, Parser)]
pub struct GenerationCli {
    #[command(flatten)]
    pub base: Cli,

    #[arg(long, default_value = "meta-llama/Meta-Llama-3.1-8B-Instruct")]
    pub model: String,

    #[arg(short, long)]
    pub do_sample: bool,

    #[arg(long, default_value = "0.6")]
    pub temperature: f64,

    #[arg(long, default_value = "0.9")]
    pub top_p: f32,

    #[arg(long, default_value = "50")]
    pub top_k: usize,

    #[arg(long)]
    pub system_prompt: Option<String>,
}

fn main() -> Result<()> {
    let args = GenerationCli::parse();

    let device = args.base.get_device()?;
    println!("Model: {}", args.model);
    println!("Device: {:?}", device);

    // Load the model and the tokenizer
    let tokenizer = AutoTokenizer::from_pretrained(args.model.clone(), None, None)?;
    let model = AutoModelForCausalLM::from_pretrained(args.model, &device, None, None)?;

    // Create a token streamer to stream the generated tokens by the model
    let (token_streamer, output_receiver) =
        TextIteratorStreamer::new(tokenizer.clone(), true, true);
    let token_streamer: Arc<Mutex<dyn TokenStreamer>> = Arc::new(Mutex::new(token_streamer));

    // Run the model generation loop in a background thread
    let (sender, receiver) = mpsc::channel::<Option<String>>();
    let generation_handle = std::thread::spawn(move || {
        let mut messages = Vec::new();

        if let Some(system_prompt) = args.system_prompt {
            messages.push(Message::system(system_prompt));
        }

        while let Ok(message) = receiver.recv() {
            if message.is_none() {
                println!("Stopping generation loop...");
                break;
            }

            let prompt = message.unwrap();
            messages.push(Message::user(prompt));

            // let prompt = tokenizer
            //     .apply_chat_template(messages.clone(), true)
            //     .unwrap();

            let mut encodings = tokenizer
                .apply_chat_template_and_encode(messages.clone(), true)
                .map_err(Error::msg)
                .unwrap();
            encodings.to_device(&device).unwrap();

            let input_ids = encodings.get_input_ids();

            let params = GenerationParams::new()
                .with_generation_config(GenerationConfig {
                    do_sample: args.do_sample,
                    top_p: Some(args.top_p),
                    top_k: Some(args.top_k),
                    temperature: args.temperature,
                    max_new_tokens: Some(2048),
                    ..GenerationConfig::default()
                })
                .with_tokenizer(tokenizer.clone())
                .with_token_streamer(token_streamer.clone());

            let output = model.generate(input_ids, params).unwrap();

            let inputs_prompt_length: usize = input_ids
                .to_vec2::<u32>()
                .unwrap()
                .first()
                .map(|seq_input_ids| tokenizer.decode(&seq_input_ids[..], true).unwrap().len())
                .unwrap_or(0);
            let sequence = &output[0].get_sequences()[0];
            let system_message: String = tokenizer
                .decode(&sequence, true)
                .unwrap()
                .chars()
                .skip(inputs_prompt_length)
                .collect();
            messages.push(Message::system(system_message));
        }
    });

    // User input loop
    loop {
        // read user input
        let mut input = String::new();
        print!("> ");
        std::io::stdout().flush().unwrap();
        std::io::stdin().read_line(&mut input).unwrap();
        let input = input.trim().to_string();

        if input.is_empty() {
            continue;
        }

        if input.to_lowercase() == "/quit" || input.to_lowercase() == "/exit" {
            sender.send(None)?;
            break;
        }

        // Send the user message to the background thread
        sender.send(Some(input))?;

        // Print the new tokens generated by the model in the background thread
        while let Ok(message) = output_receiver.recv() {
            if let Some(text) = message {
                print!("{}", text);
                io::stdout().flush().unwrap();
            } else {
                println!("");
                break;
            }
        }
    }

    generation_handle.join().unwrap();

    Ok(())
}
