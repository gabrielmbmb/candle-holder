use candle_holder_tokenizers::tokenizers::bert::BertTokenizer;

fn main() {
    let tokenizer = BertTokenizer::from_pretrained("bert-base-uncased", None).unwrap();
    println!("{:?}", tokenizer);
}
