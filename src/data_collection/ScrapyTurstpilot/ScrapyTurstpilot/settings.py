# -*- coding: utf-8 -*-

# Scrapy settings for ScrapyTurstpilot project
#
# For simplicity, this file contains only settings considered important or
# commonly used. You can find more settings consulting the documentation:
#
#     https://docs.scrapy.org/en/latest/topics/settings.html
#     https://docs.scrapy.org/en/latest/topics/downloader-middleware.html
#     https://docs.scrapy.org/en/latest/topics/spider-middleware.html

BOT_NAME = 'ScrapyTurstpilot'

SPIDER_MODULES = ['ScrapyTurstpilot.spiders']
NEWSPIDER_MODULE = 'ScrapyTurstpilot.spiders'

# Obey robots.txt rules
ROBOTSTXT_OBEY = False

# Configure maximum concurrent requests performed by Scrapy (default: 16)
CONCURRENT_REQUESTS = 32

# Configure a delay for requests for the same website (default: 0)
# See https://docs.scrapy.org/en/latest/topics/settings.html#download-delay
# See also autothrottle settings and docs
DOWNLOAD_DELAY = 0.5

FEEDS = {
    'scraped_data.csv': {
        'format': 'csv',
        'encoding': 'utf8'
    }
}

