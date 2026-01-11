"""
Twitter scraper for collecting fraud complaints and scam reports
"""

import tweepy
import pandas as pd
from typing import List, Dict
import time
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class TwitterScraper:
    """Scrape fraud-related tweets for dataset creation"""
    
    def __init__(self, api_key: str, api_secret: str, 
                 access_token: str, access_token_secret: str):
        """
        Initialize Twitter API client
        
        Args:
            api_key: Twitter API key
            api_secret: Twitter API secret
            access_token: Twitter access token
            access_token_secret: Twitter access token secret
        """
        auth = tweepy.OAuthHandler(api_key, api_secret)
        auth.set_access_token(access_token, access_token_secret)
        self.api = tweepy.API(auth, wait_on_rate_limit=True)
        
    def search_fraud_tweets(self, query: str, max_tweets: int = 1000) -> List[Dict]:
        """
        Search for fraud-related tweets
        
        Args:
            query: Search query
            max_tweets: Maximum number of tweets to fetch
            
        Returns:
            List of tweet dictionaries
        """
        tweets_data = []
        
        try:
            tweets = tweepy.Cursor(
                self.api.search_tweets,
                q=query,
                lang="en",
                tweet_mode="extended"
            ).items(max_tweets)
            
            for tweet in tweets:
                tweet_data = {
                    'id': tweet.id_str,
                    'created_at': tweet.created_at,
                    'text': tweet.full_text,
                    'user': tweet.user.screen_name,
                    'retweet_count': tweet.retweet_count,
                    'favorite_count': tweet.favorite_count,
                    'query': query
                }
                tweets_data.append(tweet_data)
                
            logger.info(f"Collected {len(tweets_data)} tweets for query: {query}")
            
        except Exception as e:
            logger.error(f"Error collecting tweets: {str(e)}")
            
        return tweets_data
    
    def collect_fraud_dataset(self, save_path: str = "data/raw/twitter/fraud_tweets.csv"):
        """
        Collect comprehensive fraud dataset from Twitter
        
        Args:
            save_path: Path to save the collected data
        """
        # Define fraud-related search queries
        queries = [
            "phone scam",
            "fraud call",
            "fake IRS call",
            "tech support scam",
            "bank fraud call",
            "social security scam",
            "robocall scam",
            "phone fraud warning",
            "scam caller",
            "phishing call"
        ]
        
        all_tweets = []
        
        for query in queries:
            logger.info(f"Searching for: {query}")
            tweets = self.search_fraud_tweets(query, max_tweets=200)
            all_tweets.extend(tweets)
            time.sleep(2)  # Rate limiting
        
        # Convert to DataFrame and save
        df = pd.DataFrame(all_tweets)
        df.drop_duplicates(subset=['text'], inplace=True)
        df.to_csv(save_path, index=False)
        
        logger.info(f"Saved {len(df)} unique tweets to {save_path}")
        return df


# Example usage
if __name__ == "__main__":
    # NOTE: Replace with your actual Twitter API credentials
    scraper = TwitterScraper(
        api_key="YOUR_API_KEY",
        api_secret="YOUR_API_SECRET",
        access_token="YOUR_ACCESS_TOKEN",
        access_token_secret="YOUR_ACCESS_TOKEN_SECRET"
    )
    
    # Collect fraud tweets
    # scraper.collect_fraud_dataset()
    print("Twitter scraper initialized. Add your credentials to use.")