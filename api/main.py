from fastapi import FastAPI
from datetime import date
from typing import Optional
import pandas as pd

app = FastAPI()

# Replace this with a database call
def load_speech_data():
    df = pd.read_csv("processed_speeches.csv")  # or from DB
    return df

@app.get("/")
def root():
    return {"message": "Welcome to the Speech API!"}

@app.get("/timeseries")
def timeseries(speaker: Optional[str] = None, institution: Optional[str] = None):
    df = load_speech_data()
    if speaker:
        df = df[df['speaker'] == speaker]
    if institution:
        df = df[df['institution'] == institution]
    return df.groupby("date")["sentiment_score"].mean().reset_index().to_dict(orient="records")
