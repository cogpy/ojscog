"""
Monitoring and Analytics Dashboard
Provides real-time metrics, performance tracking, and system health monitoring
Implements Prometheus-compatible metrics endpoint and web dashboard
"""

import logging
import time
import psutil
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from threading import Lock
from pathlib import Path
import os

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """Single metric data point"""
    timestamp: float
    value: float
    labels: Dict[str, str] = None
    
    def __post_init__(self):
        if self.labels is None:
            self.labels = {}


@dataclass
class SystemMetrics:
    """System resource metrics"""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_percent: float
    disk_used_gb: float
    disk_available_gb: float
    network_sent_mb: float
    network_recv_mb: float


@dataclass
class AgentMetrics:
    """Agent performance metrics"""
    agent_name: str
    tasks_processed: int
    tasks_successful: int
    tasks_failed: int
    avg_processing_time: float
    last_execution: str
    success_rate: float


@dataclass
class WorkflowMetrics:
    """Workflow statistics"""
    total_submissions: int
    pending_submissions: int
    in_review_submissions: int
    accepted_submissions: int
    rejected_submissions: int
    avg_review_time_days: float
    avg_quality_score: float


class MetricsCollector:
    """
    Collects and stores metrics from various sources
    Thread-safe metric collection and aggregation
    """
    
    def __init__(self, retention_hours: int = 24):
        """
        Initialize metrics collector
        
        Args:
            retention_hours: How long to retain metrics
        """
        self.retention_hours = retention_hours
        self.retention_seconds = retention_hours * 3600
        
        # Metric storage
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.timeseries: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        
        # Thread safety
        self.lock = Lock()
        
        # System metrics baseline
        self.network_baseline = psutil.net_io_counters()
    
    def increment_counter(self, name: str, value: int = 1, labels: Dict = None):
        """
        Increment a counter metric
        
        Args:
            name: Metric name
            value: Increment value
            labels: Optional labels
        """
        with self.lock:
            key = self._make_key(name, labels)
            self.counters[key] += value
    
    def set_gauge(self, name: str, value: float, labels: Dict = None):
        """
        Set a gauge metric
        
        Args:
            name: Metric name
            value: Gauge value
            labels: Optional labels
        """
        with self.lock:
            key = self._make_key(name, labels)
            self.gauges[key] = value
    
    def observe_histogram(self, name: str, value: float, labels: Dict = None):
        """
        Observe a value in histogram
        
        Args:
            name: Metric name
            value: Observed value
            labels: Optional labels
        """
        with self.lock:
            key = self._make_key(name, labels)
            self.histograms[key].append(value)
    
    def record_timeseries(self, name: str, value: float, labels: Dict = None):
        """
        Record a time series data point
        
        Args:
            name: Metric name
            value: Value
            labels: Optional labels
        """
        with self.lock:
            key = self._make_key(name, labels)
            point = MetricPoint(
                timestamp=time.time(),
                value=value,
                labels=labels or {}
            )
            self.timeseries[key].append(point)
            
            # Clean old data
            self._cleanup_timeseries(key)
    
    def _cleanup_timeseries(self, key: str):
        """Remove old time series data"""
        cutoff = time.time() - self.retention_seconds
        series = self.timeseries[key]
        
        while series and series[0].timestamp < cutoff:
            series.popleft()
    
    def _make_key(self, name: str, labels: Dict = None) -> str:
        """Create metric key with labels"""
        if not labels:
            return name
        
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"
    
    def get_counter(self, name: str, labels: Dict = None) -> int:
        """Get counter value"""
        key = self._make_key(name, labels)
        return self.counters.get(key, 0)
    
    def get_gauge(self, name: str, labels: Dict = None) -> float:
        """Get gauge value"""
        key = self._make_key(name, labels)
        return self.gauges.get(key, 0.0)
    
    def get_histogram_stats(self, name: str, labels: Dict = None) -> Dict:
        """Get histogram statistics"""
        key = self._make_key(name, labels)
        values = list(self.histograms.get(key, []))
        
        if not values:
            return {
                'count': 0,
                'sum': 0,
                'min': 0,
                'max': 0,
                'avg': 0,
                'p50': 0,
                'p95': 0,
                'p99': 0
            }
        
        sorted_values = sorted(values)
        count = len(sorted_values)
        
        return {
            'count': count,
            'sum': sum(sorted_values),
            'min': sorted_values[0],
            'max': sorted_values[-1],
            'avg': sum(sorted_values) / count,
            'p50': sorted_values[int(count * 0.50)],
            'p95': sorted_values[int(count * 0.95)],
            'p99': sorted_values[int(count * 0.99)]
        }
    
    def get_timeseries(
        self,
        name: str,
        labels: Dict = None,
        start_time: float = None,
        end_time: float = None
    ) -> List[MetricPoint]:
        """
        Get time series data
        
        Args:
            name: Metric name
            labels: Optional labels
            start_time: Start timestamp
            end_time: End timestamp
            
        Returns:
            List of metric points
        """
        key = self._make_key(name, labels)
        series = list(self.timeseries.get(key, []))
        
        if start_time:
            series = [p for p in series if p.timestamp >= start_time]
        
        if end_time:
            series = [p for p in series if p.timestamp <= end_time]
        
        return series
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        # CPU
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Memory
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_mb = memory.used / (1024 * 1024)
        memory_available_mb = memory.available / (1024 * 1024)
        
        # Disk
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        disk_used_gb = disk.used / (1024 * 1024 * 1024)
        disk_available_gb = disk.free / (1024 * 1024 * 1024)
        
        # Network
        network = psutil.net_io_counters()
        network_sent_mb = (network.bytes_sent - self.network_baseline.bytes_sent) / (1024 * 1024)
        network_recv_mb = (network.bytes_recv - self.network_baseline.bytes_recv) / (1024 * 1024)
        
        metrics = SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_mb=memory_used_mb,
            memory_available_mb=memory_available_mb,
            disk_percent=disk_percent,
            disk_used_gb=disk_used_gb,
            disk_available_gb=disk_available_gb,
            network_sent_mb=network_sent_mb,
            network_recv_mb=network_recv_mb
        )
        
        # Record as time series
        self.record_timeseries("system_cpu_percent", cpu_percent)
        self.record_timeseries("system_memory_percent", memory_percent)
        self.record_timeseries("system_disk_percent", disk_percent)
        
        return metrics


class MonitoringDashboard:
    """
    Main monitoring dashboard
    Aggregates metrics and provides dashboard interface
    """
    
    def __init__(self, metrics_collector: MetricsCollector = None):
        """
        Initialize monitoring dashboard
        
        Args:
            metrics_collector: Metrics collector instance
        """
        self.collector = metrics_collector or MetricsCollector()
        self.start_time = time.time()
    
    def get_system_status(self) -> Dict:
        """Get current system status"""
        metrics = self.collector.collect_system_metrics()
        uptime_seconds = time.time() - self.start_time
        
        return {
            'status': 'healthy' if metrics.cpu_percent < 80 and metrics.memory_percent < 80 else 'degraded',
            'uptime_seconds': uptime_seconds,
            'uptime_human': self._format_uptime(uptime_seconds),
            'system': asdict(metrics),
            'timestamp': datetime.now().isoformat()
        }
    
    def get_agent_metrics(self) -> List[AgentMetrics]:
        """Get metrics for all agents"""
        agent_names = [
            'research_discovery',
            'submission_assistant',
            'editorial_orchestration',
            'review_coordination',
            'content_quality',
            'publishing_production',
            'analytics_monitoring'
        ]
        
        metrics = []
        
        for agent in agent_names:
            labels = {'agent': agent}
            
            tasks_processed = self.collector.get_counter('agent_tasks_total', labels)
            tasks_successful = self.collector.get_counter('agent_tasks_success', labels)
            tasks_failed = self.collector.get_counter('agent_tasks_failed', labels)
            
            processing_stats = self.collector.get_histogram_stats('agent_processing_time', labels)
            
            success_rate = (tasks_successful / tasks_processed * 100) if tasks_processed > 0 else 0
            
            last_execution = self.collector.get_gauge('agent_last_execution', labels)
            last_execution_str = datetime.fromtimestamp(last_execution).isoformat() if last_execution > 0 else 'never'
            
            metrics.append(AgentMetrics(
                agent_name=agent,
                tasks_processed=tasks_processed,
                tasks_successful=tasks_successful,
                tasks_failed=tasks_failed,
                avg_processing_time=processing_stats['avg'],
                last_execution=last_execution_str,
                success_rate=success_rate
            ))
        
        return metrics
    
    def get_workflow_metrics(self) -> WorkflowMetrics:
        """Get workflow statistics"""
        return WorkflowMetrics(
            total_submissions=self.collector.get_counter('workflow_submissions_total'),
            pending_submissions=self.collector.get_gauge('workflow_submissions_pending'),
            in_review_submissions=self.collector.get_gauge('workflow_submissions_in_review'),
            accepted_submissions=self.collector.get_counter('workflow_submissions_accepted'),
            rejected_submissions=self.collector.get_counter('workflow_submissions_rejected'),
            avg_review_time_days=self.collector.get_gauge('workflow_avg_review_time_days'),
            avg_quality_score=self.collector.get_gauge('workflow_avg_quality_score')
        )
    
    def get_performance_chart_data(
        self,
        metric_name: str,
        hours: int = 24
    ) -> Dict:
        """
        Get chart data for performance metrics
        
        Args:
            metric_name: Metric to chart
            hours: Hours of history
            
        Returns:
            Chart data dictionary
        """
        start_time = time.time() - (hours * 3600)
        series = self.collector.get_timeseries(metric_name, start_time=start_time)
        
        return {
            'metric': metric_name,
            'data': [
                {
                    'timestamp': p.timestamp,
                    'datetime': datetime.fromtimestamp(p.timestamp).isoformat(),
                    'value': p.value
                }
                for p in series
            ],
            'count': len(series)
        }
    
    def get_dashboard_data(self) -> Dict:
        """Get complete dashboard data"""
        return {
            'system': self.get_system_status(),
            'agents': [asdict(m) for m in self.get_agent_metrics()],
            'workflow': asdict(self.get_workflow_metrics()),
            'charts': {
                'cpu': self.get_performance_chart_data('system_cpu_percent', hours=1),
                'memory': self.get_performance_chart_data('system_memory_percent', hours=1)
            }
        }
    
    def export_prometheus_metrics(self) -> str:
        """
        Export metrics in Prometheus format
        
        Returns:
            Prometheus-formatted metrics string
        """
        lines = []
        
        # System metrics
        system = self.collector.collect_system_metrics()
        lines.append(f"# HELP system_cpu_percent CPU usage percentage")
        lines.append(f"# TYPE system_cpu_percent gauge")
        lines.append(f"system_cpu_percent {system.cpu_percent}")
        
        lines.append(f"# HELP system_memory_percent Memory usage percentage")
        lines.append(f"# TYPE system_memory_percent gauge")
        lines.append(f"system_memory_percent {system.memory_percent}")
        
        # Agent metrics
        for agent_metric in self.get_agent_metrics():
            agent = agent_metric.agent_name
            
            lines.append(f"# HELP agent_tasks_total Total tasks processed")
            lines.append(f"# TYPE agent_tasks_total counter")
            lines.append(f'agent_tasks_total{{agent="{agent}"}} {agent_metric.tasks_processed}')
            
            lines.append(f"# HELP agent_success_rate Success rate percentage")
            lines.append(f"# TYPE agent_success_rate gauge")
            lines.append(f'agent_success_rate{{agent="{agent}"}} {agent_metric.success_rate}')
        
        return "\n".join(lines)
    
    def _format_uptime(self, seconds: float) -> str:
        """Format uptime in human-readable format"""
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        minutes = int((seconds % 3600) // 60)
        
        parts = []
        if days > 0:
            parts.append(f"{days}d")
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        
        return " ".join(parts) if parts else "< 1m"


class DashboardServer:
    """
    Simple HTTP server for monitoring dashboard
    Serves metrics endpoint and web interface
    """
    
    def __init__(
        self,
        dashboard: MonitoringDashboard,
        port: int = 9090,
        host: str = "0.0.0.0"
    ):
        """
        Initialize dashboard server
        
        Args:
            dashboard: MonitoringDashboard instance
            port: Server port
            host: Server host
        """
        self.dashboard = dashboard
        self.port = port
        self.host = host
    
    def handle_metrics(self) -> tuple:
        """Handle /metrics endpoint"""
        metrics = self.dashboard.export_prometheus_metrics()
        return (200, 'text/plain', metrics)
    
    def handle_dashboard(self) -> tuple:
        """Handle /dashboard endpoint"""
        data = self.dashboard.get_dashboard_data()
        return (200, 'application/json', json.dumps(data, indent=2))
    
    def handle_health(self) -> tuple:
        """Handle /health endpoint"""
        status = self.dashboard.get_system_status()
        return (200, 'application/json', json.dumps(status, indent=2))
    
    def generate_html_dashboard(self) -> str:
        """Generate HTML dashboard"""
        data = self.dashboard.get_dashboard_data()
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>OJS Cognitive Enhancement - Monitoring Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a1a;
            color: #e0e0e0;
            padding: 20px;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{ margin-bottom: 30px; color: #fff; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .card {{ background: #2a2a2a; border-radius: 8px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.3); }}
        .card h2 {{ font-size: 18px; margin-bottom: 15px; color: #4CAF50; }}
        .metric {{ display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #3a3a3a; }}
        .metric:last-child {{ border-bottom: none; }}
        .metric-label {{ color: #aaa; }}
        .metric-value {{ font-weight: bold; color: #fff; }}
        .status-healthy {{ color: #4CAF50; }}
        .status-degraded {{ color: #ff9800; }}
        .status-error {{ color: #f44336; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #3a3a3a; }}
        th {{ background: #333; color: #4CAF50; font-weight: 600; }}
        tr:hover {{ background: #333; }}
        .progress-bar {{ background: #3a3a3a; height: 20px; border-radius: 10px; overflow: hidden; }}
        .progress-fill {{ background: linear-gradient(90deg, #4CAF50, #8BC34A); height: 100%; transition: width 0.3s; }}
        .refresh-btn {{ background: #4CAF50; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; }}
        .refresh-btn:hover {{ background: #45a049; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 OJS Cognitive Enhancement - Monitoring Dashboard</h1>
        
        <div class="grid">
            <div class="card">
                <h2>System Status</h2>
                <div class="metric">
                    <span class="metric-label">Status</span>
                    <span class="metric-value status-{data['system']['status']}">{data['system']['status'].upper()}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Uptime</span>
                    <span class="metric-value">{data['system']['uptime_human']}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">CPU Usage</span>
                    <span class="metric-value">{data['system']['system']['cpu_percent']:.1f}%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Memory Usage</span>
                    <span class="metric-value">{data['system']['system']['memory_percent']:.1f}%</span>
                </div>
            </div>
            
            <div class="card">
                <h2>Workflow Metrics</h2>
                <div class="metric">
                    <span class="metric-label">Total Submissions</span>
                    <span class="metric-value">{data['workflow']['total_submissions']}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Pending</span>
                    <span class="metric-value">{data['workflow']['pending_submissions']}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">In Review</span>
                    <span class="metric-value">{data['workflow']['in_review_submissions']}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Avg Review Time</span>
                    <span class="metric-value">{data['workflow']['avg_review_time_days']:.1f} days</span>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>Agent Performance</h2>
            <table>
                <thead>
                    <tr>
                        <th>Agent</th>
                        <th>Tasks Processed</th>
                        <th>Success Rate</th>
                        <th>Avg Time (s)</th>
                        <th>Last Execution</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(f'''
                    <tr>
                        <td>{agent['agent_name'].replace('_', ' ').title()}</td>
                        <td>{agent['tasks_processed']}</td>
                        <td>{agent['success_rate']:.1f}%</td>
                        <td>{agent['avg_processing_time']:.2f}</td>
                        <td>{agent['last_execution']}</td>
                    </tr>
                    ''' for agent in data['agents'])}
                </tbody>
            </table>
        </div>
        
        <div style="margin-top: 20px; text-align: center;">
            <button class="refresh-btn" onclick="location.reload()">🔄 Refresh Dashboard</button>
        </div>
    </div>
    
    <script>
        // Auto-refresh every 30 seconds
        setTimeout(() => location.reload(), 30000);
    </script>
</body>
</html>
        """
        
        return html


# Global dashboard instance
_dashboard: Optional[MonitoringDashboard] = None


def get_dashboard() -> MonitoringDashboard:
    """Get global dashboard instance"""
    global _dashboard
    
    if _dashboard is None:
        _dashboard = MonitoringDashboard()
    
    return _dashboard


if __name__ == "__main__":
    # Test monitoring dashboard
    logging.basicConfig(level=logging.INFO)
    
    print("\n=== Monitoring Dashboard Test ===\n")
    
    dashboard = get_dashboard()
    
    # Simulate some metrics
    for i in range(5):
        dashboard.collector.increment_counter('workflow_submissions_total')
        dashboard.collector.increment_counter('agent_tasks_total', labels={'agent': 'research_discovery'})
        dashboard.collector.increment_counter('agent_tasks_success', labels={'agent': 'research_discovery'})
        dashboard.collector.observe_histogram('agent_processing_time', 2.5 + i * 0.5, labels={'agent': 'research_discovery'})
    
    # Get dashboard data
    data = dashboard.get_dashboard_data()
    
    print("System Status:")
    print(f"  Status: {data['system']['status']}")
    print(f"  Uptime: {data['system']['uptime_human']}")
    print(f"  CPU: {data['system']['system']['cpu_percent']:.1f}%")
    print(f"  Memory: {data['system']['system']['memory_percent']:.1f}%")
    print()
    
    print("Workflow Metrics:")
    print(f"  Total Submissions: {data['workflow']['total_submissions']}")
    print()
    
    print("Agent Metrics:")
    for agent in data['agents']:
        if agent['tasks_processed'] > 0:
            print(f"  {agent['agent_name']}: {agent['tasks_processed']} tasks, {agent['success_rate']:.1f}% success")
    print()
    
    # Export Prometheus metrics
    print("Prometheus Metrics (first 500 chars):")
    print(dashboard.export_prometheus_metrics()[:500] + "...")
