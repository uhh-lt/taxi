from jnt.isas.taxo import TaxonomyResources, TaxonomyFeatures

relations_fpath = "/Users/alex/tmp/semeval/new/relations.csv"
voc_fpath = "/Users/alex/tmp/semeval/new/voc.csv"
isa_fpath = "/Users/alex/tmp/semeval/new/en_dt.csv-isas.csv"

taxo_res = TaxonomyResources(freq_fpaths=[""], isa_fpaths=[isa_fpath])
taxo_features = TaxonomyFeatures(taxo_res, voc_fpath=voc_fpath, lang='en')
taxo_features.fill_direct_isas()