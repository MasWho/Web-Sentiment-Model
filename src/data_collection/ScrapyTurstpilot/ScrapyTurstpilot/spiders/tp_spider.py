import scrapy
from pandas import read_csv, DataFrame
import re

# The custom Spider will subclass the scrapy.Spider class
# Custom attributes and methods are then defined
class TrustPilotSpider(scrapy.Spider):
    
    # 'name' is the identifier for the Spider
    name = "trustpilot"
    
    # Get company urls from the selenium extract
    # These urls will be used as start urls to be used for the spider to crawl from
    company_urls_df = read_csv("../data/company_urls.csv")
    
    # The Spider will crawl from these Reqeusts
    # Subsequent requests will be generated successively from these initial requests
    start_urls = company_urls_df['url'].unique().tolist()[::-1]
    
            
    # A method that will be called to handle the response for each of the requests
    # The 'response' argument is an instance of TextResponse that holds the page content
    # Main function is to define how to parse the response and extracting scraped data
    # Can also contain logic for finding new URLs to follow and making new requests from them
    def parse(self, response):
        
        company_name = response.xpath('//span[@class="multi-size-header__big"]/text()').get()
        company_logo = ":".join(["https", 
                                 response.xpath('//img[@class="business-unit-profile-summary__image"]/@src').get()])
        company_website = response.xpath("//a[@class='badge-card__section badge-card__section--hoverable company_website']/@href").get()
        comments = response.xpath("//div[@class='review-content__body']")
        comments = [[c.strip() for c in comment.xpath(".//text()").getall() if len(c.strip())>0] 
                    for comment in comments]
        ratings = response.xpath("//div[@class='star-rating star-rating--medium']/img/@alt").getall()
        ratings = [int(re.match('\d+', rating).group(0)) for rating in ratings]
        
        for comment, rating in zip(comments, ratings):
            yield {
                'company_name': company_name,
                'company_logo': company_logo,
                'company_website': company_website,
                'url_website': response.url, 
                'comment': comment,
                'rating': rating
            }
            
        # This allows the spider class to recursively follow links
        # Shortcut for creating a Request object applied
        next_page = response.xpath("//a[@class='button button--primary next-page']/@href").get() 
        if next_page is not None:
            yield response.follow(next_page, callback=self.parse)