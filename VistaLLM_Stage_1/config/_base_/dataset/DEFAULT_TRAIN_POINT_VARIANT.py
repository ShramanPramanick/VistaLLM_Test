POINT_TRAIN_COMMON_CFG_LOCAL = dict(
    type='Point_QA_local',
    filename='pointQA_local_train.jsonl',
    image_folder='Visual_Genome/combined',
    template_file=r"VQA.json",
)

POINT_TRAIN_COMMON_CFG_TWICE = dict(
    type='Point_QA_twice',
    filename='pointQA_twice_train.jsonl',
    image_folder='Visual_Genome/combined',
    template_file=r"VQA.json",
)

POINT_TRAIN_COMMON_CFG_V7W = dict(
    type='V7W_POINT',
    filename='v7w_pointing_train.jsonl',
    image_folder='Visual7W/images',
    template_file=r"VQA.json",
)

DEFAULT_TRAIN_POINT_VARIANT = dict(
    POINT_LOCAL_b=dict(**POINT_TRAIN_COMMON_CFG_LOCAL, version='b'),
    POINT_LOCAL_p=dict(**POINT_TRAIN_COMMON_CFG_LOCAL, version='p'),
    POINT_LOCAL_bp=dict(**POINT_TRAIN_COMMON_CFG_LOCAL, version='bp'),
    POINT_TWICE_oq_b=dict(**POINT_TRAIN_COMMON_CFG_TWICE, version='oq-b'),
    POINT_TWICE_oq_p=dict(**POINT_TRAIN_COMMON_CFG_TWICE, version='oq-p'),
    POINT_TWICE_oq_bp=dict(**POINT_TRAIN_COMMON_CFG_TWICE, version='oq-bp'),
    POINT_TWICE_sq_b=dict(**POINT_TRAIN_COMMON_CFG_TWICE, version='sq-b'),
    POINT_TWICE_sq_p=dict(**POINT_TRAIN_COMMON_CFG_TWICE, version='sq-p'),
    POINT_TWICE_sq_bp=dict(**POINT_TRAIN_COMMON_CFG_TWICE, version='sq-bp'),
    POINT_TWICE_gq_b=dict(**POINT_TRAIN_COMMON_CFG_TWICE, version='gq-b'),
    POINT_TWICE_gq_p=dict(**POINT_TRAIN_COMMON_CFG_TWICE, version='gq-p'),
    POINT_TWICE_gq_bp=dict(**POINT_TRAIN_COMMON_CFG_TWICE, version='gq-bp'),
    POINT_V7W_p=dict(**POINT_TRAIN_COMMON_CFG_V7W, version='p'),
    POINT_V7W_b=dict(**POINT_TRAIN_COMMON_CFG_V7W, version='b'),
)