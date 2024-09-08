#[macro_export]
macro_rules! generate_router {
    ($pipeline:ident, $request:ident, $response:ident, $process_fn:expr, $warm_up_fn:expr) => {
        use anyhow::Result;
        use axum::{routing::post, Router};
        use std::sync::Arc;
        use tokio::sync::mpsc;

        use crate::cli::Cli;
        use crate::inference_endpoint::inference;
        use crate::responses::ErrorResponse;
        use crate::workers::{task_distributor, InferenceState, InferenceTask, ProcessFn};

        pub fn router(args: &Cli) -> Result<Router> {
            let model = args.model();
            let device = args.device()?;
            let dtype = args.dtype();
            let dtype_str = dtype.map_or("default".to_string(), |dt| format!("{:?}", dt));

            tracing::info!(
                "Loading {} for model '{}' on device {:?} with dtype {}",
                stringify!($pipeline),
                model,
                device,
                dtype_str
            );

            let pipeline = Arc::new($pipeline::new(&args.model(), &args.device()?, dtype, None)?);

            tracing::info!("Warming up the model...");
            $warm_up_fn(&pipeline).unwrap_or_else(|e| {
                tracing::error!("Failed to warm up the model: {}", e);
            });

            let (tx, rx) =
                mpsc::channel::<InferenceTask<$request, Result<$response, ErrorResponse>>>(32);

            tokio::spawn(task_distributor::<
                $pipeline,
                $request,
                $response,
                ProcessFn<$pipeline, $request, $response>,
            >(
                rx, pipeline, args.num_workers(), Arc::new($process_fn)
            ));

            let state = InferenceState { tx };

            Ok(Router::new().route("/", post(inference)).with_state(state))
        }
    };
}
