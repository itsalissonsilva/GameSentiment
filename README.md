# Steam Review Sentiment Analyzer

This project looks at Steam user reviews for a game and turns them into a simple sentiment dashboard.

You enter a Steam app ID, the app pulls official Steam reviews, and then it shows a few different ways of looking at how players feel about the game. The goal was to move away from fragile movie review scraping and build something that still keeps the text tied to a clear source.

## Why this version exists

This app is an update of an earlier Letterboxd based sentiment project. That version was fun to build, but scraping movie reviews became unreliable because of anti bot protections. Steam was a better fit because the reviews come from an official endpoint and each review is clearly linked to a specific game.

## Tech used

* Python
* Flask
* Steam reviews endpoint
* matplotlib
* wordcloud
* VADER sentiment
* OpenAI API for the radar style LLM analysis

## Running it locally

Install the requirements:

```bash
pip install -r requirements.txt
```

Set your OpenAI key if you want the LLM radar analysis:

```powershell
$env:OPENAI\_API\_KEY="your\_key\_here"
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Notes

* The app works best when you use a Steam app ID directly.
* The short game blurb on the results page is built from Steam store metadata.
* The LLM is used for the multidimensional radar analysis, not for the basic metadata summary.

## Example app IDs

* 1145360 for Hades
* 730 for Counter Strike 2
* 570 for Dota 2
* 1091500 for Cyberpunk 2077

## Project idea in one line

Take a game, read what players wrote, and turn that into something easier to explore.

