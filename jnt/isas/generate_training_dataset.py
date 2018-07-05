from jnt.isas.direction import load_relations, add_cohypo_negatives

taxo_en_plants_fpath = "/Users/alex/tmp/semeval/TExEval_trialdata_1.2/WN_plants.taxo"
taxo_en_vehicles_fpath = "/Users/alex/tmp/semeval/TExEval_trialdata_1.2/WN_vehicles.taxo"
taxo_en_ai_fpath = "/Users/alex/tmp/semeval/TExEval_trialdata_1.2/ontolearn_AI.taxo"
taxo_eval_en_ai_fpath = "/Users/alex/tmp/semeval/TExEval_trialdata_1.2/ontolearn_AI.taxo.eval"
relations_fpath = "/Users/alex/tmp/semeval/super/new-relations-en7.csv"
isa_fpath = "/Users/alex/tmp/semeval/super/isas-positive.csv"

relations = load_relations(relations_fpath, taxo_en_plants_fpath, taxo_en_vehicles_fpath,
               taxo_en_ai_fpath, taxo_eval_en_ai_fpath)
relations = add_cohypo_negatives(relations, isa_fpath)
relations.to_csv(relations_fpath, sep="\t", encoding="utf-8", float_format='%.0f', index=False)
print("Dataset:", relations_fpath)