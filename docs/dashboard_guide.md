### docs/dashboard_guide.md

# Dashboard User Guide

This guide provides detailed information on using the Retail Analytics Dashboard, including navigation, features, and common tasks.

## Accessing the Dashboard

The dashboard is available at:

```
http://localhost:8501
```

## Dashboard Structure

The dashboard consists of multiple pages, accessible from the sidebar:

1. **Home**: Overview and key metrics
2. **Sales Analysis**: Detailed sales trends and analysis
3. **Forecasting**: Sales predictions and model metrics
4. **Customer Segments**: Segment profiles and analysis
5. **Product Reviews**: Sentiment analysis and feature insights
6. **RAG Q&A**: Interactive product question answering

## Common Features

### Date Range Selection

Most pages include a date range selector in the sidebar:

1. Click on the date range input
2. Select start and end dates
3. Click "Apply" to update the visualizations

### Filters

Common filters available in the sidebar include:

- **Store**: Filter by specific store or view all stores
- **Category**: Filter by product category
- **Product**: Filter by specific product

Apply filters by selecting options from the dropdown menus and clicking "Apply Filters".

### Downloading Data

Most pages include options to download the displayed data:

1. Scroll to the bottom of the page
2. Click "Download CSV" or "Download Excel"
3. Save the file to your computer

### Interacting with Charts

Charts in the dashboard are interactive:

- **Hover**: View detailed information about data points
- **Zoom**: Click and drag to zoom into a specific area
- **Pan**: Hold Shift and drag to pan the view
- **Reset**: Double-click to reset the view
- **Download**: Click the camera icon to download the chart as an image

## Page Guides

### 1. Home Page

The home page provides an overview of key metrics and insights:

#### Key Performance Indicators (KPIs)

The top section displays KPIs including:
- Total Sales
- Average Transaction Value
- Customer Count
- Online Sales Percentage

#### Recent Trends

The trends section shows:
- Sales trend over the selected period
- Category performance
- Channel distribution (online vs. in-store)

#### Insights

The insights section highlights:
- Top performing products
- Underperforming categories
- Anomalies and opportunities

### 2. Sales Analysis Page

The sales analysis page provides detailed insights into sales performance:

#### Sales Trend

The sales trend chart shows:
- Daily/weekly/monthly sales over time
- Comparison with previous periods
- Trend lines and moving averages

To change the time granularity:
1. Select "Daily", "Weekly", or "Monthly" from the dropdown
2. The chart will update automatically

#### Category Analysis

The category analysis section shows:
- Sales by category
- Category growth rates
- Category contribution to total sales

To sort categories:
1. Click on the column header in the table
2. The table will sort by the selected column

#### Channel Analysis

The channel analysis section shows:
- Online vs. in-store sales
- Channel growth rates
- Channel performance by category

#### Store Comparison

The store comparison section allows you to:
- Compare sales across different stores
- Identify top and bottom performing stores
- Analyze store-specific trends

To compare stores:
1. Select stores from the multi-select dropdown
2. The comparison chart will update automatically

### 3. Forecasting Page

The forecasting page provides sales predictions and model insights:

#### Sales Forecast

The forecast chart shows:
- Historical sales data
- Predicted future sales
- Prediction intervals (best and worst case scenarios)

To adjust the forecast horizon:
1. Use the "Forecast Days" slider in the sidebar
2. The forecast will update automatically

#### Model Metrics

The model metrics section shows:
- R² Score: Goodness of fit
- RMSE: Root Mean Square Error
- MAE: Mean Absolute Error
- MAPE: Mean Absolute Percentage Error

#### Feature Importance

The feature importance chart shows:
- Which factors most influence sales
- Relative importance of each feature

#### Scenario Analysis

The scenario analysis section allows you to:
- Create "what-if" scenarios
- Adjust parameters like promotions, pricing, etc.
- See how changes might affect future sales

To create a scenario:
1. Adjust the parameters using the sliders and inputs
2. Click "Run Scenario"
3. The forecast will update with the new scenario

### 4. Customer Segments Page

The customer segments page provides insights into customer groups:

#### Segment Overview

The segment overview shows:
- Number of segments
- Size of each segment
- Key characteristics of each segment

#### Segment Profiles

The segment profiles section shows:
- Detailed characteristics of each segment
- Radar charts comparing segments across dimensions
- Key metrics for each segment

To view a specific segment:
1. Select the segment from the dropdown
2. The profile details will update automatically

#### Segment Distribution

The segment distribution section shows:
- How customers are distributed across segments
- How segments compare on key metrics

#### Recommendations

The recommendations section provides:
- Targeted marketing strategies for each segment
- Product recommendations for each segment
- Retention strategies for at-risk segments

### 5. Product Reviews Page

The product reviews page provides insights from customer reviews:

#### Sentiment Overview

The sentiment overview shows:
- Overall sentiment distribution
- Sentiment trends over time
- Average ratings

#### Feature Analysis

The feature analysis section shows:
- Sentiment by product feature
- Most mentioned features
- Feature co-occurrence patterns

To analyze a specific feature:
1. Select the feature from the dropdown
2. The feature details will update automatically

#### Review Explorer

The review explorer allows you to:
- Browse individual reviews
- Filter reviews by sentiment, rating, etc.
- Search for specific keywords

To search reviews:
1. Enter keywords in the search box
2. The reviews will filter automatically

#### Product Summaries

The product summaries section provides:
- AI-generated summaries of product reviews
- Key strengths and weaknesses
- Feature-level sentiment analysis

### 6. RAG Q&A Page

The RAG Q&A page provides an interactive interface to ask questions about products:

#### Question Input

To ask a question:
1. Type your question in the input box
2. Click "Submit" or press Enter
3. The answer will appear below

Example questions:
- "What are the best features of the TechPro X20?"
- "How does the SmartWatch Pro battery life compare to competitors?"
- "What do customers say about the HomeConnect Hub setup process?"

#### Answer Display

The answer display shows:
- AI-generated answer to your question
- Source reviews that informed the answer
- Confidence level of the answer

#### Related Products

The related products section shows:
- Products related to your question
- Comparison of related products
- Links to product details

#### Conversation History

The conversation history shows:
- Previous questions and answers
- Option to continue the conversation
- Option to start a new conversation

To clear the conversation:
1. Click "Clear Conversation"
2. The history will be reset

## Troubleshooting

### Dashboard Not Loading

If the dashboard fails to load:
1. Check that the API is running (`http://localhost:8000/api/health`)
2. Refresh the browser page
3. Check your internet connection

### Data Not Updating

If data doesn't update after changing filters:
1. Click "Apply Filters" again
2. Check that the date range is valid
3. Try refreshing the page

### Charts Not Displaying

If charts fail to display:
1. Check that JavaScript is enabled in your browser
2. Try a different browser
3. Clear your browser cache

### Error Messages

Common error messages and solutions:

- **"API Error"**: The API is not responding. Check that it's running.
- **"No Data Available"**: No data matches your filter criteria. Try broadening your filters.
- **"Date Range Error"**: The selected date range is invalid. Ensure the end date is after the start date.

## Getting Help

For additional help:
- Check the documentation in the `/docs` folder
- Contact the support team at mosesyebei@gmail.com
- Submit issues on GitHub at https://github.com/moses-y/retail-analytics/issues
