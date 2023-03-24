from fastapi import FastAPI
import uvicorn
from podcast_summarization.genism_summarizer import get_genism_summary
from nltk.tokenize import sent_tokenize

app= FastAPI()

@app.get('/PODCAST SUMMARIZER')
def podcast_summarizer(transcript):
    transcript_sentences = sent_tokenize(transcript)
    num_sentences = len(transcript_sentences)
    if num_sentences==1 and len(transcript)>30:
        return ('There is only one sentence in transcript')
    else:
        return get_genism_summary(transcript)

