"""
Communication Efficiency Tracker
Monitors and analyzes communication efficiency in federated learning
"""

import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CommunicationTracker:
    """
    Tracks communication efficiency metrics for federated learning.
    Monitors bandwidth usage, compression ratios, and communication costs.
    """
    
    def __init__(self, experiment_name: str = "default", log_dir: str = "logs"):
        """
        Initialize communication tracker.
        
        Args:
            experiment_name: Name of the experiment
            log_dir: Directory to save communication logs
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Communication metrics storage
        self.round_metrics = []
        self.client_metrics = {}
        self.global_metrics = {
            'total_bytes_transmitted': 0,
            'total_bytes_original': 0,
            'total_rounds': 0,
            'total_clients': 0,
            'compression_savings': 0.0
        }
        
        # Timing information
        self.communication_times = []
        self.round_start_times = {}
        
        logger.info(f"Initialized communication tracker for experiment: {experiment_name}")
    
    def start_round_tracking(self, round_num: int, num_clients: int):
        """Start tracking communication for a federated round."""
        self.round_start_times[round_num] = time.time()
        
        round_data = {
            'round_num': round_num,
            'num_clients': num_clients,
            'start_time': self.round_start_times[round_num],
            'client_communications': [],
            'total_bytes_sent': 0,
            'total_bytes_original': 0,
            'round_compression_ratio': 0.0
        }
        
        self.round_metrics.append(round_data)
        logger.debug(f"Started tracking round {round_num} with {num_clients} clients")
    
    def track_client_communication(self, round_num: int, client_id: str, 
                                  communication_info: Dict):
        """
        Track communication from a specific client.
        
        Args:
            round_num: Current federated learning round
            client_id: Identifier for the client
            communication_info: Dict with communication details
        """
        if not self.round_metrics or self.round_metrics[-1]['round_num'] != round_num:
            logger.warning(f"Round {round_num} not properly initialized for tracking")
            return
        
        current_round = self.round_metrics[-1]
        
        # Extract communication metrics
        bytes_sent = communication_info.get('bytes_sent', 0)
        bytes_original = communication_info.get('bytes_original', 0)
        compression_ratio = communication_info.get('compression_ratio', 1.0)
        transmission_time = communication_info.get('transmission_time', 0.0)
        network_condition = communication_info.get('network_condition', {})
        
        client_comm = {
            'client_id': client_id,
            'bytes_sent': bytes_sent,
            'bytes_original': bytes_original,
            'compression_ratio': compression_ratio,
            'transmission_time': transmission_time,
            'bandwidth_mbps': network_condition.get('bandwidth', 0),
            'latency_ms': network_condition.get('latency', 0),
            'loss_rate': network_condition.get('loss_rate', 0)
        }
        
        current_round['client_communications'].append(client_comm)
        current_round['total_bytes_sent'] += bytes_sent
        current_round['total_bytes_original'] += bytes_original
        
        # Update client-specific metrics
        if client_id not in self.client_metrics:
            self.client_metrics[client_id] = {
                'total_bytes_sent': 0,
                'total_bytes_original': 0,
                'total_rounds_participated': 0,
                'avg_compression_ratio': 0.0,
                'communication_history': []
            }
        
        client_data = self.client_metrics[client_id]
        client_data['total_bytes_sent'] += bytes_sent
        client_data['total_bytes_original'] += bytes_original
        client_data['total_rounds_participated'] += 1
        client_data['communication_history'].append(client_comm)
        
        # Update average compression ratio for client
        if client_data['total_bytes_original'] > 0:
            client_data['avg_compression_ratio'] = (
                client_data['total_bytes_sent'] / client_data['total_bytes_original']
            )
        
        logger.debug(f"Tracked communication from client {client_id}: "
                    f"{bytes_sent} bytes sent, compression: {compression_ratio:.3f}")
    
    def end_round_tracking(self, round_num: int) -> Dict:
        """
        End tracking for a federated round and calculate metrics.
        
        Args:
            round_num: Round number to end tracking for
            
        Returns:
            Dictionary with round communication metrics
        """
        if not self.round_metrics or self.round_metrics[-1]['round_num'] != round_num:
            logger.warning(f"Round {round_num} not found for ending tracking")
            return {}
        
        current_round = self.round_metrics[-1]
        end_time = time.time()
        round_duration = end_time - current_round['start_time']
        
        # Calculate round metrics
        if current_round['total_bytes_original'] > 0:
            current_round['round_compression_ratio'] = (
                current_round['total_bytes_sent'] / current_round['total_bytes_original']
            )
        
        current_round['end_time'] = end_time
        current_round['round_duration'] = round_duration
        current_round['avg_transmission_time'] = np.mean([
            comm['transmission_time'] for comm in current_round['client_communications']
        ]) if current_round['client_communications'] else 0.0
        
        # Update global metrics
        self.global_metrics['total_bytes_transmitted'] += current_round['total_bytes_sent']
        self.global_metrics['total_bytes_original'] += current_round['total_bytes_original']
        self.global_metrics['total_rounds'] += 1
        
        if self.global_metrics['total_bytes_original'] > 0:
            self.global_metrics['compression_savings'] = 1 - (
                self.global_metrics['total_bytes_transmitted'] / 
                self.global_metrics['total_bytes_original']
            )
        
        self.communication_times.append(round_duration)
        
        logger.info(f"Round {round_num} communication summary: "
                   f"{current_round['total_bytes_sent']} bytes sent, "
                   f"compression: {current_round['round_compression_ratio']:.3f}")
        
        return {
            'round_num': round_num,
            'duration': round_duration,
            'total_bytes_sent': current_round['total_bytes_sent'],
            'compression_ratio': current_round['round_compression_ratio'],
            'num_clients': len(current_round['client_communications'])
        }
    
    def get_bandwidth_efficiency(self) -> Dict:
        """Calculate bandwidth efficiency metrics."""
        if not self.round_metrics:
            return {}
        
        total_sent = sum(r['total_bytes_sent'] for r in self.round_metrics)
        total_original = sum(r['total_bytes_original'] for r in self.round_metrics)
        
        efficiency_metrics = {
            'total_bytes_sent': total_sent,
            'total_bytes_original': total_original,
            'overall_compression_ratio': total_sent / total_original if total_original > 0 else 1.0,
            'bandwidth_savings_percent': (1 - total_sent / total_original) * 100 if total_original > 0 else 0.0,
            'average_round_compression': np.mean([
                r['round_compression_ratio'] for r in self.round_metrics
            ]) if self.round_metrics else 1.0
        }
        
        return efficiency_metrics
    
    def get_communication_timeline(self) -> List[Dict]:
        """Get timeline of communication events."""
        timeline = []
        
        for round_data in self.round_metrics:
            round_event = {
                'timestamp': round_data['start_time'],
                'event_type': 'round_start',
                'round_num': round_data['round_num'],
                'num_clients': round_data['num_clients']
            }
            timeline.append(round_event)
            
            for comm in round_data['client_communications']:
                comm_event = {
                    'timestamp': round_data['start_time'] + comm['transmission_time'],
                    'event_type': 'client_communication',
                    'round_num': round_data['round_num'],
                    'client_id': comm['client_id'],
                    'bytes_sent': comm['bytes_sent'],
                    'compression_ratio': comm['compression_ratio']
                }
                timeline.append(comm_event)
            
            if 'end_time' in round_data:
                round_end_event = {
                    'timestamp': round_data['end_time'],
                    'event_type': 'round_end',
                    'round_num': round_data['round_num'],
                    'round_duration': round_data['round_duration']
                }
                timeline.append(round_end_event)
        
        return sorted(timeline, key=lambda x: x['timestamp'])
    
    def get_client_statistics(self) -> Dict:
        """Get per-client communication statistics."""
        client_stats = {}
        
        for client_id, data in self.client_metrics.items():
            stats = {
                'total_bytes_sent': data['total_bytes_sent'],
                'total_bytes_original': data['total_bytes_original'],
                'rounds_participated': data['total_rounds_participated'],
                'avg_compression_ratio': data['avg_compression_ratio'],
                'bandwidth_savings_percent': (
                    1 - data['avg_compression_ratio']
                ) * 100 if data['avg_compression_ratio'] < 1 else 0.0
            }
            
            # Calculate additional statistics from history
            if data['communication_history']:
                transmission_times = [h['transmission_time'] for h in data['communication_history']]
                stats['avg_transmission_time'] = np.mean(transmission_times)
                stats['total_transmission_time'] = sum(transmission_times)
                
                bandwidths = [h['bandwidth_mbps'] for h in data['communication_history']]
                stats['avg_bandwidth'] = np.mean(bandwidths)
            
            client_stats[client_id] = stats
        
        return client_stats
    
    def export_metrics(self, filename: Optional[str] = None) -> str:
        """
        Export communication metrics to JSON file.
        
        Args:
            filename: Optional filename. If not provided, auto-generated.
            
        Returns:
            Path to exported file
        """
        if filename is None:
            timestamp = int(time.time())
            filename = f"communication_metrics_{self.experiment_name}_{timestamp}.json"
        
        filepath = self.log_dir / filename
        
        export_data = {
            'experiment_name': self.experiment_name,
            'export_timestamp': time.time(),
            'global_metrics': self.global_metrics,
            'bandwidth_efficiency': self.get_bandwidth_efficiency(),
            'round_metrics': self.round_metrics,
            'client_statistics': self.get_client_statistics(),
            'communication_timeline': self.get_communication_timeline()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Exported communication metrics to {filepath}")
        return str(filepath)
    
    def get_summary_report(self) -> str:
        """Generate human-readable summary report."""
        if not self.round_metrics:
            return "No communication data available."
        
        efficiency = self.get_bandwidth_efficiency()
        client_stats = self.get_client_statistics()
        
        report = f"""
Communication Efficiency Report - {self.experiment_name}
{'=' * 60}

Overall Performance:
- Total Rounds: {len(self.round_metrics)}
- Total Bytes Transmitted: {efficiency['total_bytes_sent']:,}
- Original Size (uncompressed): {efficiency['total_bytes_original']:,}
- Compression Ratio: {efficiency['overall_compression_ratio']:.3f}
- Bandwidth Savings: {efficiency['bandwidth_savings_percent']:.1f}%

Client Statistics:
- Number of Clients: {len(client_stats)}
- Average Participation per Client: {np.mean([s['rounds_participated'] for s in client_stats.values()]):.1f} rounds
- Average Compression per Client: {np.mean([s['avg_compression_ratio'] for s in client_stats.values()]):.3f}

Communication Timeline:
- First Round: {min(r['start_time'] for r in self.round_metrics)}
- Last Round: {max(r.get('end_time', r['start_time']) for r in self.round_metrics)}
- Average Round Duration: {np.mean(self.communication_times):.2f} seconds
"""
        
        return report
    
    def reset_metrics(self):
        """Reset all communication metrics."""
        self.round_metrics.clear()
        self.client_metrics.clear()
        self.global_metrics = {
            'total_bytes_transmitted': 0,
            'total_bytes_original': 0,
            'total_rounds': 0,
            'total_clients': 0,
            'compression_savings': 0.0
        }
        self.communication_times.clear()
        self.round_start_times.clear()
        
        logger.info("Reset all communication metrics")
