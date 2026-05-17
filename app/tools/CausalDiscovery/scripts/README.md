# Prometheus Data Exporter

This script exports time series data from Prometheus using PromQL queries and saves it to CSV format. It handles multi-field metrics by concatenating label values with underscores and organizes data with time as the first column.

## Features

- **Multiple Query Support**: Execute multiple PromQL queries in a single run
- **Multi-field Metrics**: Automatically handles metrics with multiple labels by concatenating label values
- **Flexible Time Ranges**: Support various time formats (ISO 8601, Unix timestamps)
- **Error Handling**: Robust error handling with retry mechanisms
- **CSV Export**: Clean CSV output with time as the first column
- **Logging**: Comprehensive logging for debugging and monitoring

## Installation

Install required dependencies:

```bash
pip install pandas requests
```

## Usage

### Command Line Interface

#### Basic Usage

```bash
python scripts/prometheus_exporter.py \
    --prometheus_url http://localhost:9090 \
    --queries scripts/example_queries.json \
    --start 2023-01-01T00:00:00Z \
    --end 2023-01-02T00:00:00Z \
    --output prometheus_data.csv
```

#### With Inline Queries

```bash
python scripts/prometheus_exporter.py \
    --prometheus_url http://localhost:9090 \
    --queries '[{"name": "cpu", "promql": "rate(cpu_usage_total[5m])"}]' \
    --start 2023-01-01T00:00:00Z \
    --end 2023-01-02T00:00:00Z \
    --output prometheus_data.csv
```

#### With Custom Step and Timeout

```bash
python scripts/prometheus_exporter.py \
    --prometheus_url http://localhost:9090 \
    --queries scripts/example_queries.json \
    --start 2023-01-01T00:00:00Z \
    --end 2023-01-02T00:00:00Z \
    --step 30s \
    --timeout 60 \
    --output prometheus_data.csv \
    --verbose
```

### Programmatic Usage

```python
from scripts.prometheus_exporter import PrometheusExporter

# Create exporter
exporter = PrometheusExporter("http://localhost:9090")

# Define queries
queries = [
    {
        "name": "cpu_usage",
        "promql": "100 - (avg by (instance) (irate(node_cpu_seconds_total{mode=\"idle\"}[5m])) * 100)"
    },
    {
        "name": "memory_usage",
        "promql": "(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100"
    }
]

# Export data
df = exporter.export_data(
    queries=queries,
    start_time="2023-01-01T00:00:00Z",
    end_time="2023-01-02T00:00:00Z",
    step="1m",
    output_file="data.csv"
)
```

## Query Configuration

### JSON File Format

Create a JSON file with your queries:

```json
{
  "queries": [
    {
      "name": "cpu_usage",
      "promql": "100 - (avg by (instance) (irate(node_cpu_seconds_total{mode=\"idle\"}[5m])) * 100)"
    },
    {
      "name": "memory_usage",
      "promql": "(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100"
    },
    {
      "name": "disk_usage",
      "promql": "(1 - (node_filesystem_avail_bytes{fstype!=\"tmpfs\"} / node_filesystem_size_bytes{fstype!=\"tmpfs\"})) * 100"
    }
  ]
}
```

### Multi-field Metrics

When a query returns metrics with multiple labels, the script automatically creates separate columns:

For example, if you query:
```promql
rate(container_cpu_usage_seconds_total[5m])
```

And it returns metrics with labels like `pod="frontend-123"` and `namespace="default"`, the output will have columns like:
- `container_cpu_usage_seconds_total_frontend-123_default`
- `container_cpu_usage_seconds_total_backend-456_default`
- `container_cpu_usage_seconds_total_database-789_prod`

## Output Format

The generated CSV file has the following structure:

```csv
time,cpu_usage,memory_usage,disk_usage,container_cpu_frontend-123_default,container_cpu_backend-456_default
2023-01-01T00:00:00Z,45.2,67.8,23.1,12.5,8.3
2023-01-01T00:01:00Z,47.1,68.2,23.1,13.2,8.7
2023-01-01T00:02:00Z,46.8,67.5,23.1,12.8,8.5
...
```

- **First column**: Timestamp in ISO 8601 format
- **Subsequent columns**: One column for each metric/label combination
- **Missing values**: Empty cells for missing data points

## Time Formats

The script supports multiple time formats:

- **ISO 8601 UTC**: `2023-01-01T00:00:00Z`
- **ISO 8601 with microseconds**: `2023-01-01T00:00:00.123456Z`
- **Simple format**: `2023-01-01 00:00:00`
- **Date only**: `2023-01-01`
- **Unix timestamp**: `1672531200`

## Step Intervals

Common step intervals:

- `1m` - 1 minute
- `5m` - 5 minutes
- `15m` - 15 minutes
- `1h` - 1 hour
- `30s` - 30 seconds
- `10s` - 10 seconds

## Examples

### System Metrics

```json
{
  "queries": [
    {
      "name": "cpu_usage",
      "promql": "100 - (avg by (instance) (irate(node_cpu_seconds_total{mode=\"idle\"}[5m])) * 100)"
    },
    {
      "name": "memory_usage",
      "promql": "(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100"
    },
    {
      "name": "disk_io",
      "promql": "irate(node_disk_io_time_seconds_total[5m]) * 100"
    }
  ]
}
```

### Application Metrics

```json
{
  "queries": [
    {
      "name": "http_request_rate",
      "promql": "rate(http_requests_total[5m])"
    },
    {
      "name": "http_response_time",
      "promql": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))"
    },
    {
      "name": "http_error_rate",
      "promql": "rate(http_requests_total{status=~\"5..\"}[5m]) / rate(http_requests_total[5m]) * 100"
    }
  ]
}
```

### Container Metrics

```json
{
  "queries": [
    {
      "name": "container_cpu",
      "promql": "rate(container_cpu_usage_seconds_total[5m]) * 100"
    },
    {
      "name": "container_memory",
      "promql": "container_memory_usage_bytes / 1024 / 1024"
    },
    {
      "name": "container_network",
      "promql": "rate(container_network_receive_bytes_total[5m])"
    }
  ]
}
```

## Error Handling

The script includes comprehensive error handling:

- **Connection errors**: Automatic retry with exponential backoff
- **Invalid queries**: Clear error messages with query details
- **Time format errors**: Helpful suggestions for correct formats
- **Missing data**: Warnings for queries that return no data

## Performance Considerations

- **Large time ranges**: Use appropriate step intervals to avoid excessive data points
- **Multiple queries**: The script processes queries in sequence, not parallel
- **Memory usage**: Very large datasets may require significant memory
- **Rate limiting**: Be mindful of Prometheus rate limits

## Troubleshooting

### Common Issues

1. **Connection timeout**: Increase timeout with `--timeout` parameter
2. **No data returned**: Check PromQL syntax and time range
3. **Empty CSV**: Verify Prometheus is accessible and queries are valid
4. **Large file size**: Reduce time range or increase step interval

### Debug Mode

Enable verbose logging for debugging:

```bash
python scripts/prometheus_exporter.py \
    --prometheus_url http://localhost:9090 \
    --queries queries.json \
    --start 2023-01-01T00:00:00Z \
    --end 2023-01-02T00:00:00Z \
    --output data.csv \
    --verbose
```

## Integration with Causal Discovery

The exported CSV files can be directly used with the causal discovery system:

1. Export metrics from Prometheus using this script
2. Place the CSV file in the appropriate data directory
3. Configure the system to use the exported data
4. Run causal discovery analysis

This provides a seamless pipeline from monitoring data to causal analysis.