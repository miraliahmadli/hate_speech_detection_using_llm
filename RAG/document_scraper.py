from llama_index.readers.web import SimpleWebPageReader


def scrape_documents():
    hate_speech_explanations = [
        "https://www.coe.int/en/web/combating-hate-speech/what-is-hate-speech-and-why-is-it-a-problem-",
        "https://en.wikipedia.org/wiki/Hate_speech",
        "https://plato.stanford.edu/entries/hate-speech/",
        "https://www.un.org/en/hate-speech/understanding-hate-speech/what-is-hate-speech",
        "https://www.ala.org/advocacy/intfreedom/hate",
        "https://www.rightsforpeace.org/hate-speech",
        "https://transparency.meta.com/policies/community-standards/hate-speech/",
        "https://challengehate.com/what-is-hate-speech/",
        "https://inhope.org/EN/articles/what-is-hate-speech",
        "https://www.article19.org/resources/hate-speech-explained-a-summary/",
        "https://about.fb.com/news/2017/06/hard-questions-hate-speech/",
        "https://www.cilvektiesibugids.lv/en/themes/freedom-of-expression-media/freedom-of-expression/hate-speech/what-is-hate-speech",
        "https://items.ssrc.org/disinformation-democracy-and-conflict-prevention/classifying-and-identifying-the-intensity-of-hate-speech/"
    ]

    documents = SimpleWebPageReader(html_to_text=True).load_data(
        hate_speech_explanations
    )
    return documents
