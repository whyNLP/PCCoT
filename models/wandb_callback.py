from transformers.integrations import WandbCallback, rewrite_logs

class CustomWandbCallback(WandbCallback):
    """Read _log_cache from model and add to the log."""
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        single_value_scalars = [
            "train_runtime",
            "train_samples_per_second",
            "train_steps_per_second",
            "train_loss",
            "total_flos",
        ]

        if self._wandb is None:
            return
        if not self._initialized:
            self.setup(args, state, model)
        if state.is_world_process_zero:
            for k, v in logs.items():
                if k in single_value_scalars:
                    self._wandb.run.summary[k] = v
            non_scalar_logs = {k: v for k, v in logs.items() if k not in single_value_scalars}

            # Add custom logs from the model
            custom_log = getattr(model, "_log_cache", None)
            if isinstance(custom_log, dict):
                non_scalar_logs = {**non_scalar_logs, **custom_log}

            non_scalar_logs = rewrite_logs(non_scalar_logs)
            self._wandb.log({**non_scalar_logs, "train/global_step": state.global_step})
