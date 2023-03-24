from gensim.summarization import summarize
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt')


def get_textrank_summary(transcript):
  transcript_sentences = sent_tokenize(transcript)
  sentences = [s for s in transcript_sentences if len(s.split()) >= 7]
  filtered_transcript = ' '.join(sentences)
  return str(summarize(filtered_transcript, ratio=0.5))