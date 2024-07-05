import logging
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse

import scrapy
from scrapy.crawler import CrawlerProcess
from tqdm import tqdm

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)


def is_valid_url(url, domain, base_path):
    parsed = urlparse(url)
    return (
        parsed.scheme in ["http", "https"]
        and parsed.netloc == domain
        and parsed.path.startswith(base_path)
        and "#" not in url
    )  # Exclude URLs with fragments


def clean_url(url):
    # Replace &amp; with &, and &num; with #
    url = url.replace("&amp;", "&").replace("&num;", "#")
    # Decode URL-encoded characters
    return unquote(url)


class DocsSpider(scrapy.Spider):
    name = "docs"

    def __init__(
        self,
        homepage_url: str,
        domain: str,
        base_path: str,
        save_dir="outputs/",
        target_version=None,
        *args,
        **kwargs,
    ):
        super(DocsSpider, self).__init__(*args, **kwargs)
        self.homepage_url = homepage_url
        self.domain = domain
        self.base_path = base_path
        self.allowed_domains = [domain]
        self.start_urls = [self.homepage_url]
        self.base_dir = Path(save_dir)
        self.target_version = target_version
        self.pages = []
        self.progress_bar = None

    def start_requests(self):
        self.progress_bar = tqdm(desc="Crawling pages", unit="page")
        yield scrapy.Request(self.homepage_url, self.parse)

    def parse(self, response):
        if not is_valid_url(response.url, self.domain, self.base_path):
            return

        parsed_uri = urlparse(response.url)
        relative_path = parsed_uri.path.removeprefix(self.base_path).strip("/")
        if relative_path:
            filepath = self.base_dir / relative_path
        else:
            filepath = self.base_dir / "index.html"

        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            f.write(response.body)

        self.pages.append({"url": response.url, "html": response.body})
        self.progress_bar.update(1)

        for href in response.css("a::attr(href)").getall():
            full_url = response.urljoin(clean_url(href))
            if is_valid_url(full_url, self.domain, self.base_path):
                if self.target_version:
                    if self.target_version in full_url:
                        yield response.follow(full_url, self.parse)
                else:
                    yield response.follow(full_url, self.parse)

    def closed(self, reason):
        if self.progress_bar:
            self.progress_bar.close()


def crawl_docs(start_url, domain, base_path, save_dir="outputs/", target_version=None):
    process = CrawlerProcess(
        settings={
            "USER_AGENT": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
            "DOWNLOAD_DELAY": 2,
            "RANDOMIZE_DOWNLOAD_DELAY": True,
            "CONCURRENT_REQUESTS": 1,
            "RETRY_TIMES": 5,
            "RETRY_HTTP_CODES": [429, 500, 502, 503, 504, 522, 524, 408, 400],
            "HTTPERROR_ALLOWED_CODES": [404],  # Allow 404 errors to be logged
        }
    )

    process.crawl(
        DocsSpider,
        homepage_url=start_url,
        domain=domain,
        base_path=base_path,
        save_dir=save_dir,
        target_version=target_version,
    )
    process.start()

    spider = next(s for s in process.crawlers if s.spider.name == "docs").spider

    print(f"Total pages crawled and parsed: {len(spider.pages)}")


if __name__ == "__main__":
    # https://huggingface.co/docs/peft/v0.11.0/en/index
    # Customizable parameters
    domain = "huggingface.co"
    version = "v0.11.0"
    library = "peft"
    language = "en"

    # Construct URL and paths
    base_path = f"/docs/{library}/{version}/{language}"
    start_url = f"https://{domain}{base_path}/index"
    save_dir = f"{library}_docs_{version}"

    # Optional: Set target_version to None if you want to crawl all versions
    target_version = None

    crawl_docs(start_url, domain, base_path, save_dir, target_version)
