"""
Reddit scraper for collecting fraud complaints and scam reports
"""

import praw
import pandas as pd
from typing import List, Dict
import time
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class RedditScraper:
    """Scrape fraud-related posts from Reddit"""
    
    def __init__(self, client_id: str, client_secret: str, user_agent: str):
        """
        Initialize Reddit API client
        
        Args:
            client_id: Reddit API client ID
            client_secret: Reddit API client secret
            user_agent: User agent string
        """
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        
    def scrape_subreddit(self, subreddit_name: str, limit: int = 1000) -> List[Dict]:
        """
        Scrape posts from a specific subreddit
        
        Args:
            subreddit_name: Name of the subreddit
            limit: Maximum number of posts to fetch
            
        Returns:
            List of post dictionaries
        """
        posts_data = []
        
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            
            # Get hot, new, and top posts
            for post in subreddit.hot(limit=limit//3):
                posts_data.append(self._extract_post_data(post, subreddit_name))
            
            for post in subreddit.new(limit=limit//3):
                posts_data.append(self._extract_post_data(post, subreddit_name))
                
            for post in subreddit.top(time_filter='year', limit=limit//3):
                posts_data.append(self._extract_post_data(post, subreddit_name))
            
            logger.info(f"Collected {len(posts_data)} posts from r/{subreddit_name}")
            
        except Exception as e:
            logger.error(f"Error scraping r/{subreddit_name}: {str(e)}")
            
        return posts_data
    
    def _extract_post_data(self, post, subreddit_name: str) -> Dict:
        """Extract relevant data from a Reddit post"""
        return {
            'id': post.id,
            'title': post.title,
            'text': post.selftext,
            'subreddit': subreddit_name,
            'created_utc': datetime.fromtimestamp(post.created_utc),
            'score': post.score,
            'num_comments': post.num_comments,
            'url': post.url
        }
    
    def collect_fraud_dataset(self, save_path: str = "data/raw/reddit/fraud_posts.csv"):
        """
        Collect comprehensive fraud dataset from Reddit
        
        Args:
            save_path: Path to save the collected data
        """
        # Fraud-related subreddits
        subreddits = [
            'Scams',
            'ScamAlert',
            'Scammers',
            'phonefraud',
            'PhoneScams'
        ]
        
        all_posts = []
        
        for subreddit in subreddits:
            logger.info(f"Scraping r/{subreddit}")
            posts = self.scrape_subreddit(subreddit, limit=300)
            all_posts.extend(posts)
            time.sleep(2)  # Rate limiting
        
        # Convert to DataFrame and save
        df = pd.DataFrame(all_posts)
        df.drop_duplicates(subset=['id'], inplace=True)
        df.to_csv(save_path, index=False)
        
        logger.info(f"Saved {len(df)} unique posts to {save_path}")
        return df


# Example usage
if __name__ == "__main__":
    # NOTE: Replace with your actual Reddit API credentials
    scraper = RedditScraper(
        client_id="YOUR_CLIENT_ID",
        client_secret="YOUR_CLIENT_SECRET",
        user_agent="FraudDetector/1.0"
    )
    
    # Collect fraud posts
    # scraper.collect_fraud_dataset()
    print("Reddit scraper initialized. Add your credentials to use.")