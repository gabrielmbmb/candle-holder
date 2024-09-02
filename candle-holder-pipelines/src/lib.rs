pub mod feature_extraction;
pub mod fill_mask;
pub mod text_classification;
pub mod text_generation;
pub mod token_classification;
pub mod zero_shot_classification;

pub use feature_extraction::{FeatureExtractionOptions, FeatureExtractionPipeline, Pooling};
pub use fill_mask::{FillMaskOptions, FillMaskPipeline};
pub use text_classification::TextClassificationPipeline;
pub use text_generation::{TextGenerationPipeline, TextGenerationPipelineOutput};
pub use token_classification::{
    AggregationStrategy, TokenClassificationOptions, TokenClassificationPipeline,
};
pub use zero_shot_classification::{ZeroShotClassificationOptions, ZeroShotClassificationPipeline};
