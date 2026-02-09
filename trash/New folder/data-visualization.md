## Matplotlib
#### Barplot
```py
plt.figure(figsize=(6, 4))
plt.bar(categories, counts) # 🕷️ Using plt.plot() for categorical data, it's not appropriate
plt.title('Count of Each Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()
```

### Pie Chart
```py
plt.figure(figsize=(6,6))
plt.pie(
    Chruned_analysis,
    labels=Chruned_analysis.index,
    autopct='%1.1f%%',
    startangle=90,
    wedgeprops={'edgecolor': 'white'}
    )
plt.title("Pie Chart Showing Chruned Analysis")
plt.show()
```

### Heatmap
```py
plt.figure(figsize=(8, 10))
sns.heatmap(churn_corr, 
            annot=True, 
            fmt=".2f", 
            cmap='coolwarm', 
            cbar=True)
plt.title('Correlation of Numeric Features with Churn')
plt.show() 
```

```py
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title(f"Confusion Matrix: {model_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
```