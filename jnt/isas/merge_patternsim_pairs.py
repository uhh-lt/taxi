from pandas import read_csv, concat, DataFrame
import codecs 
from traceback import format_exc

#fpaths = ["/home/panchenko/patternsim/patternsim/output-ukwac/pairs.csv",
#          "/home/panchenko/patternsim/patternsim/output-gigaword/pairs.csv",
#          "/home/panchenko/patternsim/patternsim/output-wikipedia/pairs.csv",
#          "/home/panchenko/patternsim/patternsim/output-en-news/pairs.csv"]
fpaths = ["/home/panchenko/patternsim/PatternSim_fr/2016/wikipedia/pairs.csv",
          "/home/panchenko/patternsim/PatternSim_fr/patternsim/out_allfr_utf8/pairs.csv"]
output_fpath = "/home/panchenko/patternsim/PatternSim_fr/2016/all-pairs.csv"
start_with_num = 0
test = False 


# Load the files
dfs = {}
for fpath in fpaths:
    print("Loading patternsim pairs:", fpath)
    dfs[fpath] = read_csv(fpath, sep=";", encoding="utf-8")
    print("Loaded %d pairs" % len(dfs[fpath]))

cut_dfs_lst = []
for fpath in fpaths:
    if test: cut_dfs_lst.append(dfs[fpath][:10000])
    else: cut_dfs_lst.append(dfs[fpath])

# Merge     
cut_df = concat(cut_dfs_lst)
print("# pairs:", len(cut_df))

pair_num = 0
grouped_df = cut_df.groupby(["form","related"])
print("# of groups:", len(grouped_df))

with codecs.open(output_fpath, "w", "utf-8") as out:    
    for key, rows in grouped_df:
        try:
            pair_num += 1
            if pair_num % 10000 == 0: print(pair_num, end=' ')
            if start_with_num > pair_num: continue
            
            form, related = key
            sum_row = None
            num = 0
            for i, row in rows.iterrows():
                if num == 0: sum_row = row
                else: sum_row += row
                if pair_num == 1: 
                    print(";".join([col for col in sum_row.index.values]), file=out)
                num += 1
            sum_row.form = form
            sum_row.related = related        
            print(";".join([str(sum_row[col]) for col in sum_row.index.values]), file=out)
        except:
            print(format_exc())
    
print("Mergetd patternsim pairs:", output_fpath)
print("Merged %d pairs" % pair_num)
