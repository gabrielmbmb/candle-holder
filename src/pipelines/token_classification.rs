use std::collections::HashMap;

use anyhow::{Error, Result};
use candle_core::{Device, IndexOp, Tensor, D};
use tokenizers::Encoding;

use crate::{
    model::PreTrainedModel,
    tokenizer::{BatchEncoding, Tokenizer},
    AutoModelForTokenClassification, AutoTokenizer, FromPretrainedParameters, Padding,
};

#[derive(Debug, Clone)]
#[non_exhaustive]
struct PreEntity {
    tokens_ids: Vec<u32>,
    word: String,
    token_scores: Tensor,
    start: usize,
    end: usize,
    index: usize,
    is_subword: bool,
}

#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Entity {
    entity: String,
    score: f32,
    index: usize,
    word: String,
    start: usize,
    end: usize,
    tokens_ids: Vec<u32>,
}

impl Entity {
    pub fn new(
        entity: String,
        score: f32,
        index: usize,
        word: String,
        start: usize,
        end: usize,
        tokens_ids: Vec<u32>,
    ) -> Self {
        Self {
            entity,
            score,
            index,
            word,
            start,
            end,
            tokens_ids,
        }
    }

    pub fn get_word(&self) -> &str {
        &self.word
    }

    pub fn get_entity(&self) -> &str {
        &self.entity
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

#[derive(Debug)]
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

fn substring(s: &str, start: usize, end: usize) -> String {
    s.char_indices()
        .filter_map(|(i, c)| if i >= start && i < end { Some(c) } else { None })
        .collect()
}

pub struct TokenClassificationPipeline {
    model: Box<dyn PreTrainedModel>,
    tokenizer: Box<dyn Tokenizer>,
    device: Device,
    id2label: HashMap<usize, String>,
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
        let config = model.config();
        let id2label = config
            .id2label
            .clone()
            .ok_or_else(|| Error::msg("id2label not found in model config"))?;
        Ok(Self {
            model,
            tokenizer,
            device: device.clone(),
            id2label,
        })
    }

    fn preprocess(&mut self, inputs: Vec<String>) -> Result<BatchEncoding> {
        let mut encodings = self
            .tokenizer
            .encode(inputs, true, Some(Padding::Longest))?;
        encodings.to_device(&self.device)?;
        Ok(encodings)
    }

    // TODO: improve this function lol
    fn gather_pre_entities(
        &self,
        sentences: Vec<String>,
        batch_scores: Vec<Vec<Vec<f32>>>,
        batch_input_ids: Vec<Vec<u32>>,
        encodings: &Vec<Encoding>,
    ) -> Result<Vec<Vec<PreEntity>>> {
        let mut pre_entities: Vec<Vec<PreEntity>> = Vec::new();
        for (((sentence, scores), input_ids), encoding) in sentences
            .into_iter()
            .zip(batch_scores)
            .zip(batch_input_ids)
            .zip(encodings)
        {
            let offsets = encoding.get_offsets();
            let special_tokens_mask = encoding.get_special_tokens_mask();
            let tokens = encoding.get_tokens();
            let mut s_pre_entities: Vec<PreEntity> = Vec::new();
            for (i, token_scores) in scores
                .into_iter()
                .enumerate()
                .filter(|(i, _)| special_tokens_mask[*i] == 0)
            {
                let tokens_ids = [input_ids[i]];
                let (start, end) = offsets[i];
                let word = tokens[i].clone();
                let word_ref = substring(sentence.as_ref(), start, end);
                let is_subword = word != word_ref;
                s_pre_entities.push(PreEntity {
                    tokens_ids: tokens_ids.to_vec(),
                    word,
                    token_scores: Tensor::new(token_scores, &Device::Cpu)?,
                    start,
                    end,
                    index: i,
                    is_subword,
                });
            }
            pre_entities.push(s_pre_entities);
        }

        Ok(pre_entities)
    }

    fn aggregate_word(
        &self,
        pre_entities: Vec<PreEntity>,
        aggregation_strategy: &AggregationStrategy,
    ) -> Result<Entity> {
        let token_ids = pre_entities
            .iter()
            .flat_map(|pre_entity| pre_entity.tokens_ids.clone())
            .collect::<Vec<u32>>();
        let word = self
            .tokenizer
            .get_tokenizer()
            .decode(&token_ids[..], true)
            .map_err(Error::msg)?;
        let first_entity = pre_entities.first().unwrap().clone();
        let last_entity = pre_entities.last().unwrap().clone();
        let (index, score, entity) = match aggregation_strategy {
            AggregationStrategy::First => {
                let idx = first_entity
                    .token_scores
                    .argmax(D::Minus1)?
                    .to_scalar::<u32>()? as usize;
                let score = first_entity.token_scores.i(idx)?.to_scalar::<f32>()?;
                let entity = self.id2label.get(&idx).unwrap();
                (idx, score, entity)
            }
            AggregationStrategy::Max => {
                let max_entity = pre_entities
                    .into_iter()
                    .max_by(|a, b| {
                        let max_a = a
                            .token_scores
                            .max(D::Minus1)
                            .unwrap()
                            .to_scalar::<f32>()
                            .unwrap();
                        let max_b = b
                            .token_scores
                            .max(D::Minus1)
                            .unwrap()
                            .to_scalar::<f32>()
                            .unwrap();
                        max_a.partial_cmp(&max_b).unwrap()
                    })
                    .unwrap();
                let idx = max_entity
                    .token_scores
                    .argmax(D::Minus1)?
                    .to_scalar::<u32>()? as usize;
                let score = max_entity.token_scores.i(idx)?.to_scalar::<f32>()?;
                let entity = self.id2label.get(&idx).unwrap();
                (idx, score, entity)
            }
            AggregationStrategy::Average => {
                let average_scores = Tensor::stack(
                    &pre_entities
                        .into_iter()
                        .map(|pre_entity| pre_entity.token_scores)
                        .collect::<Vec<Tensor>>(),
                    0,
                )?
                .mean(0)?;
                let idx = average_scores.argmax(D::Minus1)?.to_scalar::<u32>()? as usize;
                let score = average_scores.i(idx)?.to_scalar::<f32>()?;
                let entity = self.id2label.get(&idx).unwrap();
                (idx, score, entity)
            }
            _ => {
                return Err(Error::msg("Aggregation strategy not implemented"));
            }
        };
        Ok(Entity::new(
            entity.clone(),
            score,
            index,
            word,
            first_entity.start,
            last_entity.end,
            token_ids,
        ))
    }

    fn aggregate_words(
        &self,
        pre_entities: Vec<Vec<PreEntity>>,
        aggregation_strategy: &AggregationStrategy,
    ) -> Result<Vec<Vec<Entity>>> {
        let mut word_entities: Vec<Vec<Entity>> = Vec::new();
        for s_pre_entities in pre_entities {
            let mut s_word_entities: Vec<Entity> = Vec::new();
            let mut word_group: Vec<PreEntity> = Vec::new();
            for pre_entity in s_pre_entities {
                if word_group.is_empty() | pre_entity.is_subword {
                    word_group.push(pre_entity);
                } else {
                    let entity = self.aggregate_word(word_group, aggregation_strategy)?;
                    s_word_entities.push(entity.clone());
                    word_group = vec![pre_entity];
                }
            }

            if !word_group.is_empty() {
                s_word_entities.push(self.aggregate_word(word_group, aggregation_strategy)?);
            }

            word_entities.push(s_word_entities);
        }

        Ok(word_entities)
    }

    fn get_tag(&self, entity_name: &str) -> (String, String) {
        if entity_name.starts_with("B-") {
            let entity = entity_name.strip_prefix("B-").unwrap().to_string();
            ("B".to_string(), entity)
        } else if entity_name.starts_with("I-") {
            let entity = entity_name.strip_prefix("I-").unwrap().to_string();
            ("I".to_string(), entity)
        } else {
            // Not B- or I- entity, default to I- for continuation
            ("I".to_string(), entity_name.to_string())
        }
    }

    fn group_sub_entities(&self, entities: Vec<Entity>) -> Result<Entity> {
        let first_entity = entities.first().unwrap();
        let last_entity = entities.last().unwrap();
        let entity = match first_entity.entity.split_once('-') {
            Some((_, entity)) => entity,
            // "O" entity
            None => &first_entity.entity,
        };
        let avg_score = entities
            .iter()
            .map(|entity| entity.score)
            .collect::<Vec<f32>>()
            .into_iter()
            .sum::<f32>()
            / entities.len() as f32;
        let token_ids = entities
            .iter()
            .flat_map(|entity| entity.tokens_ids.clone())
            .collect::<Vec<u32>>();
        let word = self
            .tokenizer
            .get_tokenizer()
            .decode(&token_ids[..], true)
            .map_err(Error::msg)?;
        Ok(Entity::new(
            entity.to_string(),
            avg_score,
            0,
            word,
            first_entity.start,
            last_entity.end,
            token_ids,
        ))
    }

    fn group_entities(&self, batch_entities: Vec<Vec<Entity>>) -> Result<Vec<Vec<Entity>>> {
        let mut batch_entity_groups: Vec<Vec<Entity>> = Vec::new();
        for entities in batch_entities {
            let mut entity_groups: Vec<Entity> = Vec::new();
            let mut entity_group_disagg: Vec<Entity> = Vec::new();
            for entity in entities {
                if entity_group_disagg.is_empty() {
                    entity_group_disagg.push(entity);
                    continue;
                }

                let (bi, tag) = self.get_tag(&entity.entity);
                let (_, last_tag) = self.get_tag(&entity_group_disagg.last().unwrap().entity);

                if tag == last_tag && bi != "B" {
                    entity_group_disagg.push(entity);
                } else {
                    entity_groups.push(self.group_sub_entities(entity_group_disagg)?);
                    entity_group_disagg = vec![entity];
                }
            }

            if !entity_group_disagg.is_empty() {
                entity_groups.push(self.group_sub_entities(entity_group_disagg)?);
            }

            batch_entity_groups.push(entity_groups);
        }

        Ok(batch_entity_groups)
    }

    fn aggregate(
        &self,
        batch_pre_entities: Vec<Vec<PreEntity>>,
        aggregation_strategy: AggregationStrategy,
    ) -> Result<Vec<Vec<Entity>>> {
        let batch_entities = match aggregation_strategy {
            AggregationStrategy::None | AggregationStrategy::Simple => {
                let mut batch_entities: Vec<Vec<Entity>> = Vec::new();
                for pre_entities in batch_pre_entities.into_iter() {
                    let mut entities: Vec<Entity> = Vec::new();
                    for pre_entity in pre_entities.into_iter() {
                        let idx = pre_entity
                            .token_scores
                            .argmax(D::Minus1)?
                            .to_scalar::<u32>()? as usize;
                        let score = pre_entity.token_scores.i(idx)?.to_scalar::<f32>()?;
                        let entity = self.id2label.get(&idx).unwrap();
                        entities.push(Entity::new(
                            entity.to_string(),
                            score,
                            pre_entity.index,
                            pre_entity.word,
                            pre_entity.start,
                            pre_entity.end,
                            pre_entity.tokens_ids,
                        ));
                    }
                    batch_entities.push(entities);
                }
                batch_entities
            }
            _ => self.aggregate_words(batch_pre_entities, &aggregation_strategy)?,
        };

        if let AggregationStrategy::None = aggregation_strategy {
            return Ok(batch_entities);
        }

        self.group_entities(batch_entities)
    }

    fn filter_entities(
        &self,
        entities: Vec<Vec<Entity>>,
        ignore_labels: Vec<String>,
    ) -> Vec<Vec<Entity>> {
        entities
            .into_iter()
            .map(|s_entities| {
                s_entities
                    .into_iter()
                    .filter(|entity| !ignore_labels.contains(&entity.entity))
                    .collect()
            })
            .collect()
    }

    fn postprocess(
        &self,
        sentences: Vec<String>,
        batch_input_ids: Vec<Vec<u32>>,
        model_outputs: &Tensor,
        encodings: &Vec<Encoding>,
        ignore_labels: Vec<String>,
        aggregation_strategy: AggregationStrategy,
    ) -> Result<Vec<Vec<Entity>>> {
        // Substract the maximum value for numerical stability
        let maxes = model_outputs.max_keepdim(D::Minus1)?;
        let shifted_exp = model_outputs.broadcast_sub(&maxes)?.exp()?;
        let batch_scores = shifted_exp
            .broadcast_div(&shifted_exp.sum_keepdim(2)?)?
            .to_vec3::<f32>()?;
        let pre_entities =
            self.gather_pre_entities(sentences, batch_scores, batch_input_ids, encodings)?;
        let entities = self.aggregate(pre_entities, aggregation_strategy)?;
        let entities = self.filter_entities(entities, ignore_labels);
        Ok(entities)
    }

    pub fn run<I: Into<String>>(
        &mut self,
        input: I,
        options: Option<TokenClassificationOptions>,
    ) -> Result<Vec<Entity>> {
        let options = options.unwrap_or_default();
        let inputs = vec![input.into()];
        let encodings = self.preprocess(inputs.clone())?;
        let output = self.model.forward(&encodings)?;
        let entities = self
            .postprocess(
                inputs,
                encodings.get_input_ids().to_vec2::<u32>()?,
                &output,
                encodings.get_encodings(),
                options.ignore_labels,
                options.aggregation_strategy,
            )?
            .first()
            .ok_or(Error::msg("No results after running postprocessing"))?
            .clone();
        Ok(entities)
    }

    pub fn run_batch<I: Into<String>>(
        &mut self,
        inputs: Vec<I>,
        options: Option<TokenClassificationOptions>,
    ) -> Result<Vec<Vec<Entity>>> {
        let options = options.unwrap_or_default();
        let inputs: Vec<String> = inputs.into_iter().map(|x| x.into()).collect();
        let encodings = self.preprocess(inputs.clone())?;
        let output = self.model.forward(&encodings)?;
        self.postprocess(
            inputs,
            encodings.get_input_ids().to_vec2::<u32>()?,
            &output,
            encodings.get_encodings(),
            options.ignore_labels,
            options.aggregation_strategy,
        )
    }
}
