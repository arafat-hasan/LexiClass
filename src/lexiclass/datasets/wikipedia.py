from __future__ import annotations

import logging
import re
import time
from typing import Dict, Iterator, Tuple, Optional
import os

logger = logging.getLogger(__name__)


def clean_wikipedia_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    text = text.strip()
    paragraphs = text.split('\n\n')
    cleaned_paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 50]
    return '\n\n'.join(cleaned_paragraphs)


def categorize_wikipedia_article(title: str, text: str) -> str:
    if not title or not text:
        return 'general'
    title_lower = title.lower()
    text_lower = text.lower()
    text_sample = text_lower[:1000]
    science_keywords = [
        'science', 'physics', 'chemistry', 'biology', 'mathematics', 'math',
        'computer', 'technology', 'engineering', 'software', 'algorithm',
        'scientific', 'research', 'laboratory', 'experiment', 'theory',
    ]
    if any(keyword in title_lower for keyword in science_keywords[:5]) or sum(keyword in text_sample for keyword in science_keywords) >= 2:
        return 'science_technology'
    history_keywords = [
        'history', 'historical', 'century', 'ancient', 'medieval', 'war',
        'empire', 'dynasty', 'battle', 'revolution', 'kingdom', 'civilization',
    ]
    if any(keyword in title_lower for keyword in history_keywords[:4]) or sum(keyword in text_sample for keyword in history_keywords) >= 2:
        return 'history'
    geo_keywords = [
        'city', 'country', 'river', 'mountain', 'geography', 'located',
        'population', 'capital', 'region', 'province', 'state', 'nation',
        'district', 'county', 'municipality',
    ]
    if any(keyword in title_lower for keyword in geo_keywords[:6]) or sum(keyword in text_sample for keyword in geo_keywords) >= 3:
        return 'geography'
    bio_indicators = [
        'born', 'died', 'birth', 'death', ' he ', ' she ', ' his ', ' her ',
        'actor', 'politician', 'writer', 'scientist', 'artist', 'musician',
    ]
    if sum(indicator in text_sample for indicator in bio_indicators) >= 3:
        return 'biography'
    sports_keywords = [
        'sport', 'team', 'player', 'game', 'championship', 'league',
        'football', 'basketball', 'baseball', 'soccer', 'tennis', 'golf',
        'olympic', 'athlete', 'coach', 'season',
    ]
    if any(keyword in title_lower for keyword in sports_keywords[:8]) or sum(keyword in text_sample for keyword in sports_keywords) >= 2:
        return 'sports'
    culture_keywords = [
        'music', 'film', 'movie', 'book', 'art', 'culture', 'museum',
        'theater', 'literature', 'novel', 'album', 'song', 'painting',
        'sculpture', 'dance', 'opera',
    ]
    if any(keyword in title_lower for keyword in culture_keywords[:8]) or sum(keyword in text_sample for keyword in culture_keywords) >= 2:
        return 'arts_culture'
    business_keywords = [
        'company', 'business', 'economy', 'economic', 'corporation',
        'industry', 'market', 'finance', 'bank', 'trade', 'commerce',
    ]
    if any(keyword in title_lower for keyword in business_keywords[:6]) or sum(keyword in text_sample for keyword in business_keywords) >= 2:
        return 'business_economics'
    return 'general'


def is_valid_article(title: str, text: str, min_length: int = 500) -> bool:
    if not title or not text:
        return False
    if 'disambiguation' in title.lower() or 'may refer to' in text.lower():
        return False
    if title.lower().startswith('list of') or title.lower().startswith('category:'):
        return False
    if text.lower().strip().startswith('#redirect'):
        return False
    if len(text) < min_length:
        return False
    alpha_ratio = sum(c.isalpha() or c.isspace() for c in text) / len(text)
    if alpha_ratio < 0.7:
        return False
    return True


def load_wikipedia_dataset(
    date: str = '20231101',
    language: str = 'en',
    max_articles: int | None = None,
    min_length: int = 500,
    subset_categories=None,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    try:
        from datasets import load_dataset

        dataset_name = f"{date}.{language}"
        logger.info("Loading Wikipedia dataset: wikimedia/wikipedia %s", dataset_name)
        start_time = time.time()
        dataset = load_dataset("wikimedia/wikipedia", dataset_name, split="train")
        logger.info("Wikipedia dataset loaded, processing articles...")
        documents: Dict[str, str] = {}
        labels: Dict[str, str] = {}
        processed_count = 0
        skipped_count = 0
        for example in dataset:
            if max_articles and processed_count >= max_articles:
                break
            title = example.get('title', '')
            text = example.get('text', '')
            cleaned_text = clean_wikipedia_text(text)
            if not is_valid_article(title, cleaned_text, min_length):
                skipped_count += 1
                continue
            category = categorize_wikipedia_article(title, cleaned_text)
            if subset_categories and category not in subset_categories:
                skipped_count += 1
                continue
            doc_id = f"wiki_{processed_count:07d}"
            documents[doc_id] = cleaned_text
            labels[doc_id] = category
            processed_count += 1
            if processed_count % 1000 == 0:
                elapsed = time.time() - start_time
                rate = processed_count / elapsed
                logger.info("Processed %d articles (%.1f articles/sec, %d skipped)", processed_count, rate, skipped_count)
        logger.info("Wikipedia processing completed: %d articles in %.2f seconds", processed_count, time.time() - start_time)
        if documents:
            label_counts: Dict[str, int] = {}
            for label in labels.values():
                label_counts[label] = label_counts.get(label, 0) + 1
            logger.info("Category distribution: %s", dict(sorted(label_counts.items())))
        return documents, labels
    except ImportError:
        logger.error("datasets library not available. Please install with: pip install datasets")
        return {}, {}
    except Exception as e:  # noqa: BLE001
        logger.error("Error loading Wikipedia dataset: %s", e)
        return {}, {}


def iter_wikipedia_dataset(
    date: str = '20231101',
    language: str = 'en',
    min_length: int = 500,
    subset_categories=None,
    max_articles: int | None = None,
) -> Iterator[Tuple[str, str, str]]:
    try:
        from datasets import load_dataset
        dataset_name = f"{date}.{language}"
        logger.info("Streaming Wikipedia dataset: wikimedia/wikipedia %s", dataset_name)
        dataset_iter = load_dataset("wikimedia/wikipedia", dataset_name, split="train", streaming=True)
        processed_count = 0
        for example in dataset_iter:
            if max_articles and processed_count >= max_articles:
                break
            title = example.get('title', '')
            text = example.get('text', '')
            cleaned_text = clean_wikipedia_text(text)
            if not is_valid_article(title, cleaned_text, min_length):
                continue
            category = categorize_wikipedia_article(title, cleaned_text)
            if subset_categories and category not in subset_categories:
                continue
            doc_id = f"wiki_{processed_count:07d}"
            processed_count += 1
            yield doc_id, cleaned_text, category
    except ImportError:
        logger.error("datasets library not available. Please install with: pip install datasets")
        return
    except Exception as e:  # noqa: BLE001
        logger.error("Error streaming Wikipedia dataset: %s", e)
        return



def iter_wikipedia_dataset_local(
    date: str = '20231101',
    language: str = 'en',
    min_length: int = 500,
    subset_categories=None,
    max_articles: Optional[int] = None,
    skip_first: int = 0,
    cache_dir: Optional[str] = None,
    offline_env: Optional[bool] = None,
) -> Iterator[Tuple[str, str, str]]:
    """Iterate Wikipedia articles from the local Hugging Face cache (no HTTP).

    - Honors `max_articles` and `skip_first` to support dataset splitting across multiple passes
      without loading all documents in memory.
    - Uses memory-mapped Arrow dataset (non-streaming) for stable, resource-safe iteration.
    """
    try:
        from datasets import load_dataset, DownloadConfig

        if offline_env is None:
            # Default to respecting the global HF offline setting if present
            offline_env = os.environ.get('HF_DATASETS_OFFLINE', '0') in {"1", "true", "TRUE", "yes", "on"}
        if offline_env:
            os.environ['HF_DATASETS_OFFLINE'] = '1'

        download_config = DownloadConfig(local_files_only=True, cache_dir=cache_dir)
        dataset_name = f"{date}.{language}"
        logger.info(
            "Loading local Wikipedia dataset (no network): wikimedia/wikipedia %s | cache_dir=%s",
            dataset_name,
            cache_dir or "<default>",
        )
        dataset = load_dataset(
            "wikimedia/wikipedia",
            dataset_name,
            split="train",
            download_config=download_config,
        )

        processed_count = 0
        yielded_count = 0
        # Iterate lazily; Arrow dataset does not load all rows into RAM.
        for example in dataset:
            title = example.get('title', '')
            text = example.get('text', '')
            cleaned_text = clean_wikipedia_text(text)
            if not is_valid_article(title, cleaned_text, min_length):
                continue
            category = categorize_wikipedia_article(title, cleaned_text)
            if subset_categories and category not in subset_categories:
                continue
            # Respect skipping for split alignment
            if processed_count < skip_first:
                processed_count += 1
                continue
            doc_id = f"wiki_{processed_count:07d}"
            processed_count += 1
            yield doc_id, cleaned_text, category
            yielded_count += 1
            if max_articles and yielded_count >= max_articles:
                break
    except ImportError:
        logger.error("datasets library not available. Please install with: pip install datasets")
        return
    except Exception as e:  # noqa: BLE001
        logger.error("Error iterating local Wikipedia dataset: %s", e)
        return


