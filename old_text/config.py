# config.py
"""
Configuration constants for BoE speech scraper.
"""
from pathlib import Path

# Base endpoints
BASE_URL = "https://www.bankofengland.co.uk"
SITEMAP_URL = f"{BASE_URL}/sitemap/speeches"
SPEECHES_URL = f"{BASE_URL}/news/speeches"
QUARTERLY_URL = "https://www.escoe.ac.uk/research/historical-data/publist/beqb/"
ARCHIVE_URL = "https://boe.access.preservica.com"

# Output directories
OUTPUT_DIR = Path("data/boe")
LOG_FILE = OUTPUT_DIR / "boe_scraper.log"

# Parsing settings
ENCODINGS = ['utf-8', 'latin-1', 'cp1252']
SPEECH_SELECTORS = [
    'a[href*="/speech/"]',
    'a[href*="/news/speeches/"]',
    # Add more specific selectors if there are other patterns for speech links on listing pages
    # Example: 'div.search-result a[href*="/speech/"]'
]
PDF_MIN_LENGTH = 200  # Minimum characters for a valid PDF extraction

# Add a default User-Agent header to mimic a web browser
DEFAULT_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36'
}

# CRITICAL: Highly specific selectors for the main content body of a speech page on bankofengland.co.uk
# These are ordered by specificity/likelihood of containing the full text.
# YOU MUST VERIFY THESE BY INSPECTING THE HTML SOURCE OF A BOE SPEECH PAGE.
TEXT_CONTENT_SELECTORS = [
    'div.rich-text',          # A common class for main content blocks on many sites
    'div.content-section',    # Another frequently used class for content areas
    'article.article-content',# If individual speeches are structured as <article> tags with this class
    'div.main-content',       # Generic main content wrapper
    'div#main-content',       # Often used as an ID for the primary content area
    'div.page-body',          # Specific body content div
    'div.body-text',          # Often used for paragraph-level text content
    'div[itemprop="articleBody"]', # Schema.org common pattern
    'main',                   # HTML5 semantic tag for main content
    'div.col-md-9',           # If content is in a Bootstrap-like grid column
    'div.col-lg-8',           # Another common grid column for main content
    'div.page-row--content', # From previous snippets, suggests rows for content
]

# CRITICAL: Selectors for "junk" content that should be REMOVED from the extracted text.
# This includes navigation, footers, share buttons, cookie banners, etc.
# YOU MUST VERIFY THESE BY INSPECTING THE HTML SOURCE OF A BOE SPEECH PAGE.
JUNK_SELECTORS = [
    'nav',                  # Navigation menus
    'footer',               # Page footers
    'header',               # Page headers
    'aside',                # Sidebars (e.g., related links)
    '.cookie-banner',       # Common class for cookie consent (seen in outputs)
    '.related-content',     # Sections like "Related articles" or "Further reading"
    '.share-buttons',       # Social media share buttons
    '.back-to-top',         # "Back to top" links
    '.print-button',        # Print functionality buttons
    '.hidden-print',        # Elements hidden in print view (often navigation)
    '.page-tools',          # Tools like "email this page", "print"
    '.page-meta',           # Metadata blocks often outside the speech text
    'form',                 # Any forms (e.g., search forms)
    'script',               # JavaScript
    'style',                # CSS
    'noscript',             # NoScript tags
    'img',                  # Images (unless image descriptions are desired as text)
    'svg',                  # SVG elements
    'iframe',               # Embedded content
    'sup',                  # Superscript (e.g., footnotes, if not desired)
    'a[href="#main-content"]', # "Skip to main content" link itself
    '.site-search',          # Site search forms
    '.site-header',          # Entire site header area
    '.site-footer',          # Entire site footer area
    '.social-media-links',   # Social media specific links
    '.search-container',     # Search specific containers
    '.breadcrumb',           # Breadcrumb navigation
    '.hero-banner'           # Top banners/heroes
]