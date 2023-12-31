use anyhow::{Error, Result};
use candle_core::{Device, Tensor, D};
use tokenizers::{EncodeInput, Offsets, Tokenizer};

use crate::{
    model::PreTrainedModel, AutoModelForTokenClassification, AutoTokenizer,
    FromPretrainedParameters,
};

#[derive(Debug)]
#[non_exhaustive]
struct PreEntity {
    word: String,
    token_scores: Vec<f32>,
    start: usize,
    end: usize,
    index: usize,
}

#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Entity {
    word: String,
    label: String,
    score: f32,
    start: usize,
    end: usize,
    index: usize,
}

impl Entity {
    pub fn new(
        word: String,
        label: String,
        score: f32,
        start: usize,
        end: usize,
        index: usize,
    ) -> Self {
        Self {
            word,
            label,
            score,
            start,
            end,
            index,
        }
    }

    pub fn get_word(&self) -> &str {
        &self.word
    }

    pub fn get_label(&self) -> &str {
        &self.label
    }

    pub fn get_score(&self) -> f32 {
        self.score
    }

    pub fn get_start(&self) -> usize {
        self.start
    }

    pub fn get_end(&self) -> usize {
        self.end
    }

    pub fn get_index(&self) -> usize {
        self.index
    }
}

pub enum AggregationStrategy {
    None,
    Simple,
    First,
    Average,
    Max,
}

pub struct TokenClassificationOptions {
    pub aggregation_strategy: AggregationStrategy,
    pub ignore_labels: Vec<String>,
}

impl Default for TokenClassificationOptions {
    fn default() -> Self {
        Self {
            aggregation_strategy: AggregationStrategy::None,
            ignore_labels: vec!["O".to_string()],
        }
    }
}

pub struct TokenClassificationPipeline {
    model: Box<dyn PreTrainedModel>,
    tokenizer: Tokenizer,
    device: Device,
}

impl TokenClassificationPipeline {
    pub fn new<S: AsRef<str> + Copy>(
        identifier: S,
        device: &Device,
        params: Option<FromPretrainedParameters>,
    ) -> Result<Self> {
        let model =
            AutoModelForTokenClassification::from_pretrained(identifier, device, params.clone())?;
        let tokenizer = AutoTokenizer::from_pretrained(identifier, params)?;
        Ok(Self {
            model,
            tokenizer,
            device: device.clone(),
        })
    }

    fn preprocessing<'s, E>(
        &self,
        inputs: Vec<E>,
    ) -> Result<(Tensor, Tensor, Vec<Vec<Offsets>>, Vec<Vec<u32>>)>
    where
        E: Into<EncodeInput<'s>> + Send,
    {
        let encodings = self
            .tokenizer
            .encode_batch(inputs, true)
            .map_err(Error::msg)?;

        let mut input_ids: Vec<Vec<u32>> = Vec::new();
        let mut token_type_ids: Vec<Vec<u32>> = Vec::new();
        let mut offsets: Vec<Vec<Offsets>> = Vec::new();
        let mut special_tokens_mask: Vec<Vec<u32>> = Vec::new();

        for encoding in encodings {
            input_ids.push(encoding.get_ids().to_vec());
            token_type_ids.push(encoding.get_type_ids().to_vec());
            offsets.push(encoding.get_offsets().to_vec());
            special_tokens_mask.push(encoding.get_special_tokens_mask().to_vec());
        }

        let input_ids = Tensor::new(input_ids, &self.device)?;
        let token_type_ids = Tensor::new(token_type_ids, &self.device)?;

        Ok((input_ids, token_type_ids, offsets, special_tokens_mask))
    }

    // TODO: improve this function lol
    fn gather_pre_entities(
        &self,
        scores: Vec<Vec<Vec<f32>>>,
        input_ids: Vec<Vec<u32>>,
        offsets: Vec<Vec<Offsets>>,
        special_tokens_mask: Vec<Vec<u32>>,
    ) -> Result<Vec<Vec<PreEntity>>> {
        let mut pre_entities: Vec<Vec<PreEntity>> = Vec::new();
        for (((s_scores, s_input_ids), s_offsets), s_special_tokens_mask) in scores
            .into_iter()
            .zip(input_ids)
            .zip(offsets)
            .zip(special_tokens_mask)
        {
            let mut s_pre_entities: Vec<PreEntity> = Vec::new();
            for (i, token_scores) in s_scores
                .into_iter()
                .enumerate()
                .filter(|(i, _)| s_special_tokens_mask[*i] == 0)
            {
                let token_input_id = s_input_ids[i];
                let word = self
                    .tokenizer
                    .decode(&[token_input_id], true)
                    .map_err(Error::msg)?;
                s_pre_entities.push(PreEntity {
                    word,
                    token_scores,
                    start: s_offsets[i].0,
                    end: s_offsets[i].1,
                    index: i,
                });
            }
            pre_entities.push(s_pre_entities);
        }

        Ok(pre_entities)
    }

    fn aggregate(
        &self,
        pre_entities: Vec<Vec<PreEntity>>,
        ignore_labels: Vec<String>,
    ) -> Result<Vec<Vec<Entity>>> {
        let config = self.model.config();
        // TODO: move this to `new` method to avoid cloning every time?
        let id2label = config
            .id2label
            .ok_or_else(|| Error::msg("id2label not found in model config"))?;

        let mut entities: Vec<Vec<Entity>> = Vec::new();
        for s_pre_entities in pre_entities.into_iter() {
            let mut s_entities: Vec<Entity> = Vec::new();
            for pre_entity in s_pre_entities.into_iter() {
                let (label_idx, score) = pre_entity
                    .token_scores
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, score)| (i, score))
                    .unwrap();
                let label = id2label.get(&label_idx.to_string()).unwrap();
                if ignore_labels.contains(label) {
                    continue;
                }
                s_entities.push(Entity::new(
                    pre_entity.word,
                    label.to_string(),
                    *score,
                    pre_entity.start,
                    pre_entity.end,
                    pre_entity.index,
                ));
            }
            entities.push(s_entities);
        }

        Ok(entities)
    }

    fn postprocessing(
        &self,
        input_ids: Vec<Vec<u32>>,
        model_outputs: &Tensor,
        offsets: Vec<Vec<Offsets>>,
        special_tokens_mask: Vec<Vec<u32>>,
        ignore_labels: Vec<String>,
    ) -> Result<Vec<Vec<Entity>>> {
        // Substract the maximum value for numerical stability
        let maxes = model_outputs.max_keepdim(D::Minus1)?;
        let shifted_exp = model_outputs.broadcast_sub(&maxes)?.exp()?;
        let scores = shifted_exp
            .broadcast_div(&shifted_exp.sum_keepdim(2)?)?
            .to_vec3::<f32>()?;
        let pre_entities =
            self.gather_pre_entities(scores, input_ids, offsets, special_tokens_mask)?;
        let entities = self.aggregate(pre_entities, ignore_labels)?;
        Ok(entities)
    }

    pub fn run<'s, E>(
        &self,
        input: E,
        options: Option<TokenClassificationOptions>,
    ) -> Result<Vec<Entity>>
    where
        E: Into<EncodeInput<'s>> + Send,
    {
        let options = options.unwrap_or_default();
        let (input_ids, token_type_ids, offsets, special_tokens_mask) =
            self.preprocessing(vec![input])?;
        let output = self.model.forward(&input_ids, &token_type_ids)?;
        let entities = self.postprocessing(
            input_ids.to_vec2::<u32>()?,
            &output,
            offsets,
            special_tokens_mask,
            options.ignore_labels,
        )?[0]
            .clone();
        Ok(entities)
    }

    pub fn run_batch<'s, E>(
        &self,
        inputs: Vec<E>,
        options: Option<TokenClassificationOptions>,
    ) -> Result<Vec<Vec<Entity>>>
    where
        E: Into<EncodeInput<'s>> + Send,
    {
        let options = options.unwrap_or_default();
        let (input_ids, token_type_ids, offsets, special_tokens_mask) =
            self.preprocessing(inputs)?;
        let output = self.model.forward(&input_ids, &token_type_ids)?;
        self.postprocessing(
            input_ids.to_vec2::<u32>()?,
            &output,
            offsets,
            special_tokens_mask,
            options.ignore_labels,
        )
    }
}
