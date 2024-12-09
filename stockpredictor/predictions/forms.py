from django import forms

class StockInputForm(forms.Form):
    COMPANIES = [
        ('AAPL', 'Apple'),
        ('GOOGL', 'Alphabet (Google)'),
        ('MSFT', 'Microsoft'),
        ('AMZN', 'Amazon'),
        # Add more companies here as needed
    ]
    
    company = forms.ChoiceField(choices=COMPANIES, label='Select Company', required=True)
    open_price = forms.FloatField(label='Open Price', required=True)
    high_price = forms.FloatField(label='High Price', required=True)
    low_price = forms.FloatField(label='Low Price', required=True)
