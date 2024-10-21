import pandas as pd
from unidecode import unidecode
from sklearn.utils import shuffle

# Function to clean and normalize text
def clean_text(text):
    text = text.replace('\n\n', '\n').strip()
    return unidecode(text)

df = pd.read_csv('dataset/movies.csv', low_memory=False)

print(len(df))

selected_cols = ["name","description"]
df = df[selected_cols]
df.dropna(axis=0, inplace=True)
print(len(df))

train_size = int(0.9 * len(df))
df = shuffle(df, random_state=7331)


with open('dataset/movie_descriptions.txt', 'w', encoding="utf-8") as f:
    for row in df.itertuples():
        if clean_text(row.description) != "No description found": 
            f.write(f"The description of the movie named '{row.name}' is:\n")
            f.write(clean_text(row.description) + "\nEND_OF_MOVIE\n")
    
print("Dataset is Done")


with open('dataset/movie_descriptions_small.txt', 'w', encoding="utf-8") as f_small:
    count = 0
    for row in df.itertuples():
        if count >= 100000:
            break
        if clean_text(row.description) != "No description found": 
            f_small.write(f"The description of the movie named '{row.name}' is:\n")
            f_small.write(clean_text(row.description) + "\nEND_OF_MOVIE\n")
            count += 1

print("Smaller subset (100,000 movies) is Done")

