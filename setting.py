# 可以使用"."来访问字典的key
class DottableDict(dict):
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.__dict__ = self
    def allowDotting(self, state=True):
        if state:
            self.__dict__ = self
        else:
            self.__dict__ = dict()

args = {}
# 构图
args["kq"] = 5
args["k"] = 5
# 损失
args["alpha"] = 1
args["beta"] = None
args["beta_percentile"] = 98
# 模型参数
args["seed"] = None
args["epochs"] = 1000
args["hidden_units"] = [2048]
args["lr"] = 0.0001
args["init_weights"] = 1e-5
args["regularizer_scale"] = 1e-5
args["layer_decay"] = 0.3
# 其他
args["dataset"] = "Oxford5k" # ['Oxford5k', 'Paris6k']
args["data_path"] = "./drive/My Drive/Datasets"

# 自定义
# args["norm_ax"] = False
# args["norm_wx"] = False
args["query_num"] = 70
args["random"] = False
args["dropout"] = 0
args["evaluate_way"] = 1 # 1本方法，0为revisited的方法
args["hparam_tuning"] = False
args["pre_graph"] = True # True使用基于空间校正的构图法，False使用基于点乘的构图法
args["norm_tv"] = 1e-5
args["attention"] = False
args["head_num"] = 8
args["train_noise"] = 0
args["record_all"] = False # 记录GCN每一层

args = DottableDict(args)

