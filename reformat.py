import pandas as pd 
import glob
  
input_path = "/Volumes/Backup/Research/Data/KHOJASTEH/Lagrangian_tracks_sub_domain_1/"
output_path = "/Volumes/Backup/Research/Data/KHOJASTEH/Lagrangian_tracks_sub_domain_1/Reformat_Subset/"

files = glob.glob(input_path+"*.txt")

for i in range(len(files)):
    print(i)

    df = pd.read_csv(files[i], 
                    sep=" ",
                    header=1) 
    
    df = df.loc[0:1000,:]

    # print(df)
    df.to_csv(output_path+"LPT_position_t_"+"{:03d}".format(i)+".csv",
              header=None,
              index=False)

# print(df)
# print(df.columns.values)
