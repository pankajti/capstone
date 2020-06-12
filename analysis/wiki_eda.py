import pandas as pd
import os
root_dir = r'/Users/pankaj/Library/Mobile Documents/com~apple~CloudDocs/Capstone/Wikipedia Data'
#root_dir = r'/Users/pankaj/dev/git/smu/capstone/data/wikipedia/Wikipedia Data'
comments_file_path = os.path.join(root_dir , 'attack_annotated_comments.tsv')
annot_file_path = os.path.join(root_dir , 'attack_annotations.tsv')

from pandas_profiling import ProfileReport


comments = pd.read_table(comments_file_path)
annotations = pd.read_table(annot_file_path)
grouped_annot = annotations.groupby('rev_id').sum()
grouped_annot = grouped_annot.reset_index()
merged_comments = comments.merge(grouped_annot, on = 'rev_id')
result_path = annot_file_path = os.path.join(root_dir , 'comments_with_grouped_annoptations.tsv')

merged_comments.to_csv(result_path, sep = '\t')
print (merged_comments)