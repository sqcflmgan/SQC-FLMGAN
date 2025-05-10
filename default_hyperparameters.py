import numpy as np

hp_default = {"local_iterations" :5, "batch_size" : 32, "weight_decay" : 0.0, "optimizer" : "SGD", "momentum" : 0.9,
              "log_frequency" : -100, "aggregation" : "mean", 
              "count_bits" : True, "participation_rate" : 1.0, "balancedness" : 1.0}

hp_net_dict = {

  'cnn': 
          {'type' : 'CNN', 'lr' : 0.01, 'batch_size' : 32, 'weight_decay' : 0.0, 'optimizer' : 'SGD', 'momentum' : 0.9,
          'lr_decay' : ['LambdaLR', {'lr_lambda' : lambda epoch: 1.0}], 'iterations' : 8000},# 'lr' : 0.125

}

def get_hp_compression(compression):

  c = compression[0]
  hp = compression[1]

  if c ==  "none" : 
    return  {"compression_up" : ["none", {}], "compression_down" : ["none", {}],
               "accumulation_up" : False, "accumulation_down" : False,  "aggregation" : "mean"}

  if c ==  "signsgd" : 
    return  {"compression_up" : ["signsgd", {}], "compression_down" : ["none", {}],
               "accumulation_up" : False, "accumulation_down" : False,  "aggregation" : "majority", "lr" : hp["lr"], "local_iterations" : 1}

  if c ==  "dgc_up" : 
    return  {"compression_up" : ["dgc", {"p" : hp["p_up"]}], "compression_down" : ["none", {}],
               "accumulation_up" : True, "accumulation_down" : False,  "aggregation" : "mean"}

  if c ==  "stc_up" : 
    return  {"compression_up" : ["stc", {"p" : hp["p_up"]}], "compression_down" : ["none", {}],
               "accumulation_up" : True, "accumulation_down" : False,  "aggregation" : "mean"}

  if c ==  "dgc_updown" : 
    return  {"compression_up" : ["dgc", {"p" : hp["p_up"]}], "compression_down" : ["dgc", {"p" : hp["p_down"]}],
               "accumulation_up" : True, "accumulation_down" : True,  "aggregation" : "mean"}    

  if c ==  "stc_updown" : 
    return {"compression_up" : ["stc", {"p" : hp["p_up"]}], "compression_down" : ["stc", {"p" : hp["p_down"]}],
               "accumulation_up" : True, "accumulation_down" : True,  "aggregation" : "mean"}

  if c ==  "fed_avg" : 
    return {"compression_up" : ["none", {}], "compression_down" : ["none", {}],
               "accumulation_up" : False, "accumulation_down" : False,  "aggregation" : "weighted_mean", "local_iterations" : hp["n"]}


def get_hp(hp_experiment):
                      
  if "multi" in hp_experiment:
    hp_experiment.update(hp_experiment["multi"])
    del hp_experiment["multi"]            

  hp = hp_default
  hp.update(hp_net_dict[hp_experiment["net"]])
  
  if "compression" in hp_experiment:
    hp_compression = get_hp_compression(hp_experiment["compression"])
    hp.update(hp_compression)
    hp.update({key : hp_experiment["compression"][1][key] for key in hp if key in hp_experiment["compression"][1]})
    del hp_experiment["compression"]

  hp.update(hp_experiment)

  hp["communication_rounds"] = hp["iterations"]//hp["local_iterations"]

  if hp["log_frequency"] < 0:
      hp['log_frequency'] = np.ceil(hp['communication_rounds']/(-hp["log_frequency"])).astype('int') 

  return hp