
def init_neptune(args, api_key, project_name, experiment_name, experiment_tags=[]):
    import neptune
    from pytorch_lightning.loggers.neptune import NeptuneLogger

    params = vars(args)

    neptune.init(
        project_qualified_name=project_name,
        api_token=api_key
    )

    neptune_logger = NeptuneLogger(
        api_key=api_key,
        project_name=project_name,
        experiment_name=experiment_name,
        tags=experiment_tags,
        params=params
    )
    return neptune_logger