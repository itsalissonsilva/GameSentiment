from __future__ import annotations

import base64
import io
import json
import os
import re
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from flask import Flask, render_template, request
from wordcloud import STOPWORDS, WordCloud

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

APP_DIR = Path(__file__).resolve().parent
CACHE_DIR = APP_DIR / 'cache'
CACHE_DIR.mkdir(exist_ok=True)

app = Flask(__name__)

FEATURED_GAMES = [
    {'appid': 1145360, 'name': 'Hades'},
    {'appid': 730, 'name': 'Counter-Strike 2'},
    {'appid': 570, 'name': 'Dota 2'},
    {'appid': 1091500, 'name': 'Cyberpunk 2077'},
    {'appid': 1172470, 'name': 'Apex Legends'},
    {'appid': 271590, 'name': 'Grand Theft Auto V'},
]

LANGUAGE_OPTIONS = ['english', 'all', 'brazilian', 'spanish', 'german', 'russian']
FILTER_OPTIONS = ['recent', 'updated', 'all']
REVIEW_TYPE_OPTIONS = ['all', 'positive', 'negative']
PURCHASE_TYPE_OPTIONS = ['steam', 'all', 'non_steam_purchase']
DEFAULT_OPENAI_MODEL = 'gpt-4.1-mini'

RADAR_AXES = [
    ('enjoyment', 'Enjoyment'),
    ('frustration', 'Frustration'),
    ('immersion', 'Immersion'),
    ('value_satisfaction', 'Value satisfaction'),
    ('technical_satisfaction', 'Technical satisfaction'),
]

STOPWORDS_EXTRA = {
    'game', 'games', 'steam', 'review', 'reviews', 'player', 'players', 'people', 'thing', 'things',
    'one', 'two', 'still', 'really', 'very', 'much', 'many', 'lot', 'lots', 'also', 'even', 'just',
    'get', 'got', 'make', 'made', 'say', 'said', 'see', 'well', 'now', 'back', 'first', 'second',
    'good', 'great', 'bad', 'fun', 'best', 'worst', 'amazing', 'awesome', 'nice', 'cool', 'okay',
    'ok', 'yes', 'yeah', 'no', 'dont', 'doesnt', 'didnt', 'cant', 'couldnt', 'wouldnt', 'isnt',
    'wasnt', 'youre', 'theyre', 'im', 'ive', 'ill', 'id', 'thats', 'theres', 'stuff', 'something',
    'anything', 'everything', 'someone', 'anyone', 'etc', 'lol', 'lmao', 'haha', 'hrs', 'hour',
    'hours', 'time', 'times', 'day', 'days', 'week', 'weeks', 'month', 'months', 'year', 'years',
    'play', 'played', 'playing', 'replay', 'gameplay', 'recommend', 'recommended', 'recommendation',
    'buy', 'bought', 'purchase', 'purchased', 'price', 'priced', 'worth', 'dlc', 'dev', 'devs',
    'developer', 'developers', 'update', 'updates', 'patch', 'patches', 'content', 'story', 'mode',
    'modes', 'match', 'matches', 'team', 'teams', 'enemy', 'enemies', 'level', 'levels', 'mission',
    'missions', 'map', 'maps', 'character', 'characters', 'weapon', 'weapons', 'item', 'items',
    'online', 'offline', 'server', 'servers', 'system', 'systems', 'u', 'ur', 'ya', 'tho', 'though',
    'could', 'would', 'should', 'might', 'must', 'ever', 'always', 'never', 'every', 'without',
    'within', 'around', 'another', 'actually', 'basically', 'pretty', 'quite', 'overall', 'mostly',
    'kind', 'sort', 'let', 'lets', 'maybe', 'probably', 'definitely', 'absolutely', 'literally',
    'fucking', 'fuck', 'shit', 'damn', 'wtf', 'bro', 'dude', 'man', 'men', 'woman', 'women'
}
NEGATIVE_HINTS = {
    'bad', 'broken', 'bug', 'bugs', 'buggy', 'boring', 'grind', 'grindy', 'crash', 'crashes', 'lag',
    'laggy', 'toxic', 'expensive', 'microtransactions', 'cheaters', 'cheater', 'refund', 'stutter',
    'issue', 'issues', 'problem', 'problems', 'annoying', 'hate', 'worse', 'worst', 'trash', 'awful',
    'terrible', 'desync', 'disconnect', 'broken', 'unbalanced', 'imbalance', 'paywall'
}
POSITIVE_HINTS = {
    'great', 'love', 'amazing', 'fantastic', 'excellent', 'smooth', 'beautiful', 'masterpiece',
    'enjoy', 'rewarding', 'favorite', 'addictive', 'solid', 'polished', 'satisfying', 'fun', 'best',
    'fair', 'stable', 'responsive', 'engaging', 'charming', 'memorable', 'strong'
}


def ensure_vader() -> None:
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon', quiet=True)



def get_vader() -> SentimentIntensityAnalyzer:
    ensure_vader()
    return SentimentIntensityAnalyzer()



def steam_headers() -> dict[str, str]:
    return {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) SteamSentimentApp/3.0',
        'Accept': 'application/json,text/plain,*/*',
    }



def resolve_appid(query: str) -> int:
    query = query.strip()
    if not query:
        raise ValueError('Enter a Steam app ID.')
    if not query.isdigit():
        raise ValueError('Use a numeric Steam app ID.')
    return int(query)



def fetch_store_details(appid: int) -> dict[str, Any]:
    url = 'https://store.steampowered.com/api/appdetails'
    response = requests.get(url, params={'appids': appid, 'l': 'english'}, headers=steam_headers(), timeout=30)
    response.raise_for_status()
    payload = response.json().get(str(appid), {})
    data = payload.get('data') or {}
    return data


def build_game_blurb(store_data: dict[str, Any], fallback_name: str) -> str:
    name = store_data.get('name') or fallback_name
    release_date = (store_data.get('release_date') or {}).get('date') or ''
    genres = ', '.join(g.get('description', '') for g in (store_data.get('genres') or []) if g.get('description'))
    developers = ', '.join(store_data.get('developers') or [])
    short_description = re.sub(r'\s+', ' ', str(store_data.get('short_description') or '')).strip()

    opening_parts = [f'{name} is a game']
    if release_date:
        opening_parts.append(f'released on {release_date}')
    if genres:
        opening_parts.append(f'in the {genres} genre' if ',' not in genres else f'in the genres {genres}')
    opening = ' '.join(opening_parts).strip() + '.'

    details = []
    if developers:
        details.append(f'Developed by {developers}.')
    if short_description:
        details.append(short_description)

    return ' '.join([opening] + details).strip()



def fetch_reviews(
    appid: int,
    language: str = 'english',
    max_reviews: int = 100,
    filter_mode: str = 'recent',
    review_type: str = 'all',
    purchase_type: str = 'steam',
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    url = f'https://store.steampowered.com/appreviews/{appid}'
    cursor = '*'
    reviews: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}

    while len(reviews) < max_reviews:
        params = {
            'json': 1,
            'language': language,
            'filter': filter_mode,
            'review_type': review_type,
            'purchase_type': purchase_type,
            'num_per_page': min(100, max_reviews - len(reviews)),
            'cursor': cursor,
        }
        response = requests.get(url, params=params, headers=steam_headers(), timeout=45)
        response.raise_for_status()
        payload = response.json()

        if payload.get('success') != 1:
            raise RuntimeError('Steam reviews API returned an unsuccessful response.')

        if not summary:
            summary = payload.get('query_summary', {}) or {}

        batch = payload.get('reviews', []) or []
        if not batch:
            break

        reviews.extend(batch)
        next_cursor = payload.get('cursor')
        if not next_cursor or next_cursor == cursor:
            break
        cursor = next_cursor

    return reviews[:max_reviews], summary



def reviews_to_dataframe(reviews: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    vader = get_vader()

    for review in reviews:
        text = (review.get('review') or '').strip()
        if not text:
            continue

        author = review.get('author') or {}
        compound = float(vader.polarity_scores(text)['compound'])
        if compound >= 0.05:
            text_label = 'positive'
        elif compound <= -0.05:
            text_label = 'negative'
        else:
            text_label = 'neutral'

        rows.append(
            {
                'recommendationid': review.get('recommendationid'),
                'review': text,
                'steam_recommendation': 'positive' if review.get('voted_up') else 'negative',
                'text_sentiment': text_label,
                'compound': compound,
                'language': review.get('language'),
                'votes_up': int(review.get('votes_up', 0) or 0),
                'votes_funny': int(review.get('votes_funny', 0) or 0),
                'comment_count': int(review.get('comment_count', 0) or 0),
                'steam_purchase': bool(review.get('steam_purchase')),
                'received_for_free': bool(review.get('received_for_free')),
                'early_access': bool(review.get('written_during_early_access')),
                'hours_at_review': round((author.get('playtime_at_review') or 0) / 60.0, 1),
                'hours_forever': round((author.get('playtime_forever') or 0) / 60.0, 1),
                'num_games_owned': int(author.get('num_games_owned', 0) or 0),
                'num_reviews_by_author': int(author.get('num_reviews', 0) or 0),
                'created': pd.to_datetime(review.get('timestamp_created', 0), unit='s', utc=True, errors='coerce'),
                'review_length': len(text),
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values('created', ascending=False).reset_index(drop=True)



def fig_to_base64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=170)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')



def make_bar_chart(series: pd.Series, title: str, xlabel: str) -> str | None:
    counts = series.value_counts().sort_index()
    if counts.empty:
        return None
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Count')
    for idx, value in enumerate(counts.values):
        ax.text(idx, value, str(int(value)), ha='center', va='bottom', fontsize=9)
    return fig_to_base64(fig)



def make_distribution_with_gaussian(series: pd.Series) -> tuple[str | None, dict[str, float]]:
    clean = pd.Series(series).dropna().astype(float)
    total_n = len(clean)
    clean = clean[~np.isclose(clean, 0.0)]
    zero_removed = total_n - len(clean)
    if clean.empty:
        return None, {'mu': 0.0, 'sigma': 0.0, 'n': 0, 'zero_removed': zero_removed, 'zero_share': 100.0 if total_n else 0.0}

    mu = float(clean.mean())
    sigma = float(clean.std(ddof=0))
    sigma = sigma if sigma > 1e-6 else 1e-6

    fig, ax = plt.subplots(figsize=(7.3, 4.9))
    ax.hist(clean, bins=18, density=True, alpha=0.78)
    x = np.linspace(max(-1.0, clean.min() - 0.05), min(1.0, clean.max() + 0.05), 500)
    gaussian = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    ax.plot(x, gaussian, linewidth=2.2)
    ax.axvline(mu, linestyle='--', linewidth=1.4)
    ax.set_title('Sentiment distribution after removing zero scores')
    ax.set_xlabel('VADER compound score')
    ax.set_ylabel('Density')
    ax.text(
        0.98,
        0.95,
        f'μ = {mu:.3f}\nσ = {sigma:.3f}\nn = {len(clean)}\nzeros removed = {zero_removed}',
        transform=ax.transAxes,
        ha='right',
        va='top',
        bbox={'boxstyle': 'round,pad=0.35', 'facecolor': 'white', 'alpha': 0.82, 'edgecolor': '#cccccc'},
        fontsize=9,
    )
    return fig_to_base64(fig), {
        'mu': mu,
        'sigma': sigma,
        'n': int(len(clean)),
        'zero_removed': int(zero_removed),
        'zero_share': float((zero_removed / total_n) * 100) if total_n else 0.0,
    }



def make_hours_scatter(df: pd.DataFrame) -> str | None:
    if df.empty:
        return None
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    x = df['hours_at_review'].clip(upper=300)
    y = df['compound']
    ax.scatter(x, y, alpha=0.65)
    ax.set_title('Hours at review vs sentiment')
    ax.set_xlabel('Hours at review (capped at 300)')
    ax.set_ylabel('VADER compound score')
    return fig_to_base64(fig)



def make_review_length_boxplot(df: pd.DataFrame) -> str | None:
    if df.empty:
        return None
    labels = [label for label in ['negative', 'neutral', 'positive'] if not df.loc[df['text_sentiment'] == label, 'review_length'].dropna().empty]
    if not labels:
        return None
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.boxplot([df.loc[df['text_sentiment'] == label, 'review_length'].dropna() for label in labels], tick_labels=labels)
    ax.set_title('Review length by text sentiment')
    ax.set_xlabel('Text sentiment')
    ax.set_ylabel('Characters')
    return fig_to_base64(fig)



def make_helpful_votes_boxplot(df: pd.DataFrame) -> str | None:
    if df.empty:
        return None
    labels = [label for label in ['negative', 'neutral', 'positive'] if not df.loc[df['text_sentiment'] == label, 'votes_up'].dropna().empty]
    if not labels:
        return None
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.boxplot([df.loc[df['text_sentiment'] == label, 'votes_up'].clip(upper=200).dropna() for label in labels], tick_labels=labels)
    ax.set_title('Helpful votes by text sentiment')
    ax.set_xlabel('Text sentiment')
    ax.set_ylabel('Helpful votes (capped at 200)')
    return fig_to_base64(fig)



def clean_word_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\b\w{1,2}\b', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text



def title_tokens(resolved_name: str) -> set[str]:
    return {token for token in re.findall(r'[a-z]{3,}', resolved_name.lower())}



def make_wordcloud(texts: list[str], title: str, resolved_name: str) -> str | None:
    joined = ' '.join(clean_word_text(t) for t in texts).strip()
    if not joined:
        return None
    stopwords = set(STOPWORDS).union(STOPWORDS_EXTRA).union(title_tokens(resolved_name))
    wc = WordCloud(
        width=1100,
        height=500,
        background_color='white',
        stopwords=stopwords,
        collocations=False,
        min_word_length=4,
        max_words=120,
    ).generate(joined)
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title)
    return fig_to_base64(fig)



def select_positive_texts(df: pd.DataFrame) -> list[str]:
    subset = df[(df['text_sentiment'] == 'positive') | ((df['steam_recommendation'] == 'positive') & (df['compound'] > 0))]
    return subset['review'].tolist()



def select_negative_texts(df: pd.DataFrame) -> list[str]:
    subset = df[(df['text_sentiment'] == 'negative') | ((df['steam_recommendation'] == 'negative') & (df['compound'] < 0))]
    return subset['review'].tolist()



def heuristic_radar_scores(df: pd.DataFrame) -> dict[str, float]:
    texts = df['review'].tolist()
    if not texts:
        return {key: 5.0 for key, _ in RADAR_AXES}

    joined = ' '.join(clean_word_text(t) for t in texts)
    words = joined.split()
    total = max(len(words), 1)
    positive_share = float((df['text_sentiment'] == 'positive').mean())
    negative_share = float((df['text_sentiment'] == 'negative').mean())
    avg_compound = float(df['compound'].mean())
    high_hours_share = float((df['hours_at_review'] >= 20).mean())

    def freq(hints: set[str]) -> float:
        return sum(word in hints for word in words) / total

    enjoyment = np.clip(5 + 6 * avg_compound + 10 * positive_share + 30 * freq(POSITIVE_HINTS), 0, 10)
    frustration = np.clip(2 + 12 * negative_share + 55 * freq(NEGATIVE_HINTS), 0, 10)
    immersion = np.clip(3 + 5 * high_hours_share + 2 * positive_share, 0, 10)
    value_satisfaction = np.clip(4 + 5 * positive_share - 3 * negative_share + 35 * freq({'worth', 'cheap', 'fair', 'priced', 'value'}), 0, 10)
    technical_satisfaction = np.clip(7 + 20 * freq({'smooth', 'stable', 'polished'}) - 60 * freq({'bug', 'bugs', 'crash', 'crashes', 'lag', 'stutter', 'server', 'servers'}) - 3 * negative_share, 0, 10)

    return {
        'enjoyment': float(enjoyment),
        'frustration': float(frustration),
        'immersion': float(immersion),
        'value_satisfaction': float(value_satisfaction),
        'technical_satisfaction': float(technical_satisfaction),
    }



def get_openai_client(api_key: str | None) -> Any:
    key = (api_key or '').strip() or os.environ.get('OPENAI_API_KEY', '').strip()
    if not key:
        return None
    if OpenAI is None:
        raise RuntimeError('openai package is not installed. Add it to requirements and reinstall dependencies.')
    return OpenAI(api_key=key)



def build_llm_prompt(df: pd.DataFrame, resolved_name: str) -> str:
    sample_df = df[['review', 'steam_recommendation', 'compound', 'hours_at_review', 'votes_up']].copy()
    sample_df = sample_df.sort_values(['votes_up', 'hours_at_review'], ascending=False).head(30)
    lines = []
    for idx, row in enumerate(sample_df.itertuples(index=False), start=1):
        text = re.sub(r'\s+', ' ', str(row.review)).strip()[:650]
        lines.append(
            f"{idx}. steam={row.steam_recommendation}; compound={row.compound:.3f}; hours={row.hours_at_review}; helpful={row.votes_up}; text={text}"
        )
    joined = '\n'.join(lines)
    return (
        f"You are analyzing Steam user reviews for the game {resolved_name}.\n"
        "Score the sampled reviews on five dimensions from 0 to 10, where higher numbers mean the feeling is more strongly present across the sample.\n"
        "Dimensions:\n"
        "- enjoyment: how much players sound like they are having fun\n"
        "- frustration: how much irritation, anger, or exhaustion is present\n"
        "- immersion: how absorbing, sticky, or hard-to-put-down the game feels\n"
        "- value_satisfaction: whether players feel the game is worth the money or time\n"
        "- technical_satisfaction: how stable, polished, and trustworthy the technical experience seems\n"
        "Use only the supplied sample. Return concise notes and numeric scores.\n\n"
        f"Sampled reviews:\n{joined}"
    )



def llm_radar_analysis(df: pd.DataFrame, resolved_name: str, api_key: str | None, model: str | None) -> tuple[dict[str, float], str, str]:
    fallback_scores = heuristic_radar_scores(df)
    chosen_model = (model or '').strip() or os.environ.get('OPENAI_MODEL', '').strip() or DEFAULT_OPENAI_MODEL
    client = get_openai_client(api_key)
    if client is None:
        return fallback_scores, 'Heuristic fallback used because no OpenAI API key was provided.', 'heuristic'

    schema = {
        'type': 'object',
        'properties': {
            'scores': {
                'type': 'object',
                'properties': {key: {'type': 'number', 'minimum': 0, 'maximum': 10} for key, _ in RADAR_AXES},
                'required': [key for key, _ in RADAR_AXES],
                'additionalProperties': False,
            },
            'notes': {
                'type': 'array',
                'items': {'type': 'string'},
                'minItems': 3,
                'maxItems': 5,
            },
            'sample_size': {'type': 'integer', 'minimum': 1},
        },
        'required': ['scores', 'notes', 'sample_size'],
        'additionalProperties': False,
    }

    try:
        response = client.responses.create(
            model=chosen_model,
            input=[
                {'role': 'system', 'content': 'Return only structured JSON matching the supplied schema.'},
                {'role': 'user', 'content': build_llm_prompt(df, resolved_name)},
            ],
            temperature=0.2,
            max_output_tokens=700,
            text={
                'format': {
                    'type': 'json_schema',
                    'name': 'steam_review_radar',
                    'schema': schema,
                    'strict': True,
                }
            },
        )
        payload = json.loads(response.output_text)
        scores = {key: float(payload['scores'][key]) for key, _ in RADAR_AXES}
        notes = ' · '.join(str(x) for x in payload.get('notes', []) if str(x).strip())
        meta = f"LLM sample size: {payload.get('sample_size', min(len(df), 30))}"
        return scores, f"{notes} {meta}".strip(), chosen_model
    except Exception as exc:
        return fallback_scores, f'Heuristic fallback used because the LLM analysis failed: {exc}', 'heuristic'



def make_radar_plot(scores: dict[str, float], title: str) -> str:
    labels = [label for _, label in RADAR_AXES]
    values = [float(scores.get(key, 0.0)) for key, _ in RADAR_AXES]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values_cycle = values + values[:1]
    angles_cycle = angles + angles[:1]

    fig = plt.figure(figsize=(7.3, 4.9))
    ax = plt.subplot(111, polar=True)
    ax.set_position([0.12, 0.16, 0.76, 0.72])
    ax.plot(angles_cycle, values_cycle, linewidth=2.2)
    ax.fill(angles_cycle, values_cycle, alpha=0.22)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_ylim(0, 10)
    ax.set_title(title, pad=16)

    for angle, value in zip(angles, values):
        ax.text(angle, min(value + 0.8, 10.2), f'{value:.1f}', ha='center', va='center', fontsize=9)

    return fig_to_base64(fig)



def prepare_meta_pills(df: pd.DataFrame, query_summary: dict[str, Any], resolved_name: str, appid: int, dist_stats: dict[str, float]) -> list[str]:
    steam_split = (df['steam_recommendation'] == 'positive').mean() * 100
    text_positive = (df['text_sentiment'] == 'positive').mean() * 100
    avg_compound = float(df['compound'].mean())
    median_hours = float(df['hours_at_review'].median())
    pills = [
        resolved_name,
        f'App ID {appid}',
        f'{len(df)} reviews analyzed',
        f'Steam positive {steam_split:.1f}%',
        f'Text positive {text_positive:.1f}%',
        f'Avg compound {avg_compound:.3f}',
        f'Median hours {median_hours:.1f} h',
        f"Zero share {dist_stats.get('zero_share', 0.0):.1f}%",
    ]
    if query_summary.get('review_score_desc'):
        pills.append(str(query_summary['review_score_desc']))
    return pills


@app.route('/', methods=['GET'])
def index():
    return render_template(
        'index.html',
        featured_games=FEATURED_GAMES,
        language_options=LANGUAGE_OPTIONS,
        filter_options=FILTER_OPTIONS,
        review_type_options=REVIEW_TYPE_OPTIONS,
        purchase_type_options=PURCHASE_TYPE_OPTIONS,
        default_openai_model=DEFAULT_OPENAI_MODEL,
    )


@app.route('/analyze', methods=['POST'])
def analyze():
    query = request.form.get('query', '').strip()
    language = request.form.get('language', 'english').strip() or 'english'
    filter_mode = request.form.get('filter_mode', 'recent').strip() or 'recent'
    review_type = request.form.get('review_type', 'all').strip() or 'all'
    purchase_type = request.form.get('purchase_type', 'steam').strip() or 'steam'
    api_key = request.form.get('openai_api_key', '').strip()
    openai_model = request.form.get('openai_model', '').strip() or DEFAULT_OPENAI_MODEL

    max_reviews = int(request.form.get('max_reviews', 100) or 100)
    max_reviews = max(20, min(max_reviews, 300))

    try:
        appid = resolve_appid(query)
        store_data = fetch_store_details(appid)
        resolved_name = store_data.get('name') or f'App {appid}'
        game_blurb = build_game_blurb(store_data, resolved_name)
        reviews, query_summary = fetch_reviews(
            appid=appid,
            language=language,
            max_reviews=max_reviews,
            filter_mode=filter_mode,
            review_type=review_type,
            purchase_type=purchase_type,
        )
        df = reviews_to_dataframe(reviews)

        if df.empty:
            return render_template('result.html', error='Steam returned no written reviews for those settings.', query=query)

        distribution_chart, dist_stats = make_distribution_with_gaussian(df['compound'])
        radar_scores, radar_note, radar_model = llm_radar_analysis(df, resolved_name, api_key, openai_model)
        radar_chart = make_radar_plot(
            radar_scores,
            'LLM sentiment radar',
        )

        positive_cloud = make_wordcloud(select_positive_texts(df), 'Positive review language', resolved_name)
        negative_cloud = make_wordcloud(select_negative_texts(df), 'Negative review language', resolved_name)

        other_charts = [
            ('Steam recommendations', make_bar_chart(df['steam_recommendation'], 'Steam recommendation split', 'Steam label')),
            ('Text sentiment labels', make_bar_chart(df['text_sentiment'], 'Text sentiment split', 'Text label')),
            ('Hours vs sentiment', make_hours_scatter(df)),
            ('Review length by sentiment', make_review_length_boxplot(df)),
        ]
        if other_charts[-1][1] is None:
            other_charts[-1] = ('Helpful votes by sentiment', make_helpful_votes_boxplot(df))

        meta_pills = prepare_meta_pills(df, query_summary, resolved_name, appid, dist_stats)
        sample_reviews = []
        for row in df[['created', 'steam_recommendation', 'text_sentiment', 'compound', 'hours_at_review', 'votes_up', 'review']].head(12).to_dict(orient='records'):
            created = row.get('created')
            if pd.notna(created):
                row['created'] = pd.Timestamp(created).strftime('%Y-%m-%d')
            else:
                row['created'] = 'Unknown date'
            sample_reviews.append(row)

        return render_template(
            'result.html',
            query=query,
            resolved_name=resolved_name,
            game_blurb=game_blurb,
            appid=appid,
            meta_pills=meta_pills,
            distribution_chart=distribution_chart,
            radar_chart=radar_chart,
            positive_cloud=positive_cloud,
            negative_cloud=negative_cloud,
            other_charts=other_charts,
            radar_note=radar_note,
            radar_model=radar_model,
            dist_stats=dist_stats,
            sample_reviews=sample_reviews,
        )
    except Exception as exc:
        return render_template('result.html', error=str(exc), query=query)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
