"""
Synthetic conversation generator for fraud and legitimate calls
"""

import random
from typing import List, Dict
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    """Generate synthetic fraud and legitimate call conversations"""
    
    def __init__(self):
        self.fraud_templates = self._load_fraud_templates()
        self.legitimate_templates = self._load_legitimate_templates()
        
    def _load_fraud_templates(self) -> List[Dict]:
        """Load fraud conversation templates"""
        return [
            {
                'type': 'IRS_SCAM',
                'template': [
                    "Hello, this is {name} from the Internal Revenue Service.",
                    "We have detected suspicious activity on your tax account.",
                    "You owe {amount} in back taxes and penalties.",
                    "If you don't pay immediately, we will issue an arrest warrant.",
                    "Can you provide your social security number to verify your identity?",
                    "We need payment via wire transfer or gift cards right away."
                ]
            },
            {
                'type': 'TECH_SUPPORT_SCAM',
                'template': [
                    "Hello, this is {name} from {company} technical support.",
                    "We've detected a virus on your computer.",
                    "Your system has been compromised and needs immediate attention.",
                    "I need remote access to your computer to fix this issue.",
                    "Can you go to this website and download our security software?",
                    "We'll need your credit card to charge for the repair service."
                ]
            },
            {
                'type': 'BANK_FRAUD',
                'template': [
                    "This is {name} from {bank} fraud department.",
                    "We've detected unauthorized charges on your account.",
                    "Your account has been temporarily suspended for your protection.",
                    "To verify your identity, I need your account number and PIN.",
                    "We also need to confirm your social security number.",
                    "Please act quickly before the fraudulent charges go through."
                ]
            },
            {
                'type': 'SOCIAL_SECURITY_SCAM',
                'template': [
                    "This is an urgent call from the Social Security Administration.",
                    "Your social security number has been suspended.",
                    "There's been suspicious activity associated with your number.",
                    "We need to verify your information immediately.",
                    "Can you confirm your social security number?",
                    "If you don't respond, legal action will be taken against you."
                ]
            },
            {
                'type': 'LOTTERY_SCAM',
                'template': [
                    "Congratulations! You've won {amount} in our national lottery.",
                    "You've been selected as one of our lucky winners.",
                    "To claim your prize, we need some information from you.",
                    "There's a small processing fee of {fee} to release your winnings.",
                    "Can you provide your bank account details for the transfer?",
                    "This offer is only valid for the next 24 hours."
                ]
            }
        ]
    
    def _load_legitimate_templates(self) -> List[Dict]:
        """Load legitimate conversation templates"""
        return [
            {
                'type': 'CUSTOMER_SERVICE',
                'template': [
                    "Hello, this is {name} from {company} customer service.",
                    "I'm calling about your recent inquiry regarding {topic}.",
                    "How can I assist you today?",
                    "I'll need to verify your account. Can you provide your account number?",
                    "Thank you for that information. Let me pull up your account.",
                    "Is there anything else I can help you with?"
                ]
            },
            {
                'type': 'APPOINTMENT_REMINDER',
                'template': [
                    "Hi, this is {name} from {company}.",
                    "I'm calling to remind you about your appointment on {date}.",
                    "Your appointment is scheduled for {time}.",
                    "Please call us if you need to reschedule.",
                    "Do you have any questions about your upcoming appointment?",
                    "We look forward to seeing you."
                ]
            },
            {
                'type': 'SURVEY',
                'template': [
                    "Hello, this is {name} from {company}.",
                    "We're conducting a brief customer satisfaction survey.",
                    "Would you have a few minutes to answer some questions?",
                    "On a scale of 1 to 10, how would you rate our service?",
                    "What could we do to improve your experience?",
                    "Thank you for your valuable feedback."
                ]
            },
            {
                'type': 'DELIVERY_NOTIFICATION',
                'template': [
                    "Hi, this is {name} from {company} delivery service.",
                    "I'm calling about your package delivery.",
                    "Your package is scheduled for delivery on {date}.",
                    "Will someone be available to receive it?",
                    "If not, we can arrange an alternative delivery time.",
                    "Thank you and have a great day."
                ]
            }
        ]
    
    def generate_conversation(self, is_fraud: bool = True) -> Dict:
        """
        Generate a single conversation
        
        Args:
            is_fraud: Whether to generate a fraud or legitimate conversation
            
        Returns:
            Dictionary containing conversation details
        """
        templates = self.fraud_templates if is_fraud else self.legitimate_templates
        template = random.choice(templates)
        
        # Fill in template variables
        conversation = []
        for line in template['template']:
            filled_line = line.format(
                name=random.choice(['John Smith', 'Sarah Johnson', 'Michael Brown', 'Lisa Davis']),
                company=random.choice(['Microsoft', 'Amazon', 'Apple', 'Google', 'Bank of America']),
                bank=random.choice(['Chase', 'Wells Fargo', 'Bank of America', 'Citibank']),
                amount=f"${random.randint(1000, 50000)}",
                fee=f"${random.randint(50, 500)}",
                topic=random.choice(['billing', 'account', 'service', 'product']),
                date=random.choice(['tomorrow', 'next Monday', 'this Friday']),
                time=random.choice(['10:00 AM', '2:00 PM', '3:30 PM'])
            )
            conversation.append(filled_line)
        
        return {
            'conversation': ' '.join(conversation),
            'type': template['type'],
            'is_fraud': 1 if is_fraud else 0,
            'label': 'fraud' if is_fraud else 'legitimate'
        }
    
    def generate_dataset(self, num_fraud: int = 500, num_legitimate: int = 500,
                        save_path: str = "data/raw/synthetic/conversations.csv") -> pd.DataFrame:
        """
        Generate a complete synthetic dataset
        
        Args:
            num_fraud: Number of fraud conversations to generate
            num_legitimate: Number of legitimate conversations to generate
            save_path: Path to save the dataset
            
        Returns:
            DataFrame containing the generated conversations
        """
        conversations = []
        
        # Generate fraud conversations
        logger.info(f"Generating {num_fraud} fraud conversations...")
        for _ in range(num_fraud):
            conversations.append(self.generate_conversation(is_fraud=True))
        
        # Generate legitimate conversations
        logger.info(f"Generating {num_legitimate} legitimate conversations...")
        for _ in range(num_legitimate):
            conversations.append(self.generate_conversation(is_fraud=False))
        
        # Create DataFrame and shuffle
        df = pd.DataFrame(conversations)
        df = df.sample(frac=1).reset_index(drop=True)
        
        # Save to CSV
        df.to_csv(save_path, index=False)
        logger.info(f"Saved {len(df)} conversations to {save_path}")
        
        return df


# Example usage
if __name__ == "__main__":
    generator = SyntheticDataGenerator()
    
    # Generate a single conversation
    fraud_conv = generator.generate_conversation(is_fraud=True)
    print("Fraud Conversation:")
    print(fraud_conv['conversation'])
    print()
    
    legitimate_conv = generator.generate_conversation(is_fraud=False)
    print("Legitimate Conversation:")
    print(legitimate_conv['conversation'])
    print()
    
    # Generate full dataset
    df = generator.generate_dataset(num_fraud=100, num_legitimate=100)
    print(f"\nGenerated dataset with {len(df)} conversations")
    print(df.head())