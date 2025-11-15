"""
message_bus.py - Asynchronous Message Bus for Agent Communication

Implements publish-subscribe pattern for inter-agent communication,
supporting parallel tensor thread fibers in the cognitive architecture.
"""

import asyncio
import json
import logging
from typing import Dict, Callable, Any, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MessagePriority(Enum):
    """Message priority levels"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


class MessageType(Enum):
    """Standard message types for agent communication"""
    # Workflow events
    SUBMISSION_RECEIVED = "submission.received"
    SUBMISSION_VALIDATED = "submission.validated"
    REVIEW_ASSIGNED = "review.assigned"
    REVIEW_COMPLETED = "review.completed"
    DECISION_MADE = "decision.made"
    PUBLICATION_READY = "publication.ready"
    
    # Agent coordination
    TASK_REQUEST = "task.request"
    TASK_RESPONSE = "task.response"
    TASK_COMPLETE = "task.complete"
    TASK_FAILED = "task.failed"
    
    # State synchronization
    STATE_UPDATE = "state.update"
    STATE_QUERY = "state.query"
    STATE_RESPONSE = "state.response"
    
    # Learning and optimization
    LEARNING_UPDATE = "learning.update"
    PERFORMANCE_METRIC = "performance.metric"
    OPTIMIZATION_SUGGESTION = "optimization.suggestion"


@dataclass
class AgentMessage:
    """Message structure for inter-agent communication"""
    sender: str
    receiver: str  # Can be specific agent or "*" for broadcast
    message_type: str
    payload: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    priority: MessagePriority = MessagePriority.NORMAL
    reply_to: Optional[str] = None
    ttl: int = 300  # Time to live in seconds
    
    def to_json(self) -> str:
        """Serialize message to JSON"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['priority'] = self.priority.value
        return json.dumps(data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'AgentMessage':
        """Deserialize message from JSON"""
        data = json.loads(json_str)
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['priority'] = MessagePriority(data['priority'])
        return cls(**data)


class MessageBus:
    """
    Asynchronous message bus for agent communication.
    
    Implements publish-subscribe pattern with:
    - Priority-based message routing
    - Message persistence and replay
    - Dead letter queue for failed messages
    - Performance monitoring
    """
    
    def __init__(self, max_queue_size: int = 10000):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.message_queue: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=max_queue_size)
        self.dead_letter_queue: List[AgentMessage] = []
        self.message_history: List[AgentMessage] = []
        self.running = False
        self.stats = {
            'messages_sent': 0,
            'messages_delivered': 0,
            'messages_failed': 0,
            'average_latency': 0.0
        }
        
    async def publish(self, message: AgentMessage) -> bool:
        """
        Publish message to bus.
        
        Args:
            message: AgentMessage to publish
            
        Returns:
            bool: True if message was queued successfully
        """
        try:
            # Add to priority queue (lower priority value = higher priority)
            await self.message_queue.put((message.priority.value, message))
            self.stats['messages_sent'] += 1
            self.message_history.append(message)
            
            logger.info(f"Published message: {message.message_type} from {message.sender} to {message.receiver}")
            return True
            
        except asyncio.QueueFull:
            logger.error(f"Message queue full, dropping message: {message.correlation_id}")
            self.dead_letter_queue.append(message)
            self.stats['messages_failed'] += 1
            return False
            
    async def subscribe(self, message_type: str, callback: Callable, agent_id: str = None):
        """
        Subscribe to message type.
        
        Args:
            message_type: Type of message to subscribe to (or "*" for all)
            callback: Async function to call when message received
            agent_id: Optional agent ID for targeted subscriptions
        """
        subscription_key = f"{message_type}:{agent_id}" if agent_id else message_type
        
        if subscription_key not in self.subscribers:
            self.subscribers[subscription_key] = []
            
        self.subscribers[subscription_key].append(callback)
        logger.info(f"Subscribed to {subscription_key}")
        
    async def unsubscribe(self, message_type: str, callback: Callable, agent_id: str = None):
        """Unsubscribe from message type"""
        subscription_key = f"{message_type}:{agent_id}" if agent_id else message_type
        
        if subscription_key in self.subscribers:
            self.subscribers[subscription_key].remove(callback)
            
    async def start(self):
        """Start message bus processing"""
        self.running = True
        logger.info("Message bus started")
        
        while self.running:
            try:
                # Get message from priority queue
                priority, message = await asyncio.wait_for(
                    self.message_queue.get(),
                    timeout=1.0
                )
                
                # Check TTL
                age = (datetime.now() - message.timestamp).total_seconds()
                if age > message.ttl:
                    logger.warning(f"Message expired: {message.correlation_id}")
                    self.dead_letter_queue.append(message)
                    continue
                
                # Route message
                await self._route_message(message)
                
            except asyncio.TimeoutError:
                # No messages in queue, continue
                continue
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                
    async def stop(self):
        """Stop message bus processing"""
        self.running = False
        logger.info("Message bus stopped")
        
    async def _route_message(self, message: AgentMessage):
        """
        Route message to subscribers.
        
        Implements parallel tensor thread fibers by dispatching
        to multiple subscribers concurrently.
        """
        start_time = datetime.now()
        callbacks = []
        
        # Find matching subscribers
        # 1. Specific agent subscriptions
        specific_key = f"{message.message_type}:{message.receiver}"
        if specific_key in self.subscribers:
            callbacks.extend(self.subscribers[specific_key])
            
        # 2. General type subscriptions
        if message.message_type in self.subscribers:
            callbacks.extend(self.subscribers[message.message_type])
            
        # 3. Wildcard subscriptions
        if "*" in self.subscribers:
            callbacks.extend(self.subscribers["*"])
            
        if not callbacks:
            logger.warning(f"No subscribers for message: {message.message_type}")
            self.dead_letter_queue.append(message)
            return
            
        # Execute callbacks concurrently (parallel processing)
        try:
            tasks = [self._execute_callback(callback, message) for callback in callbacks]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Check for failures
            failures = [r for r in results if isinstance(r, Exception)]
            if failures:
                logger.error(f"Callback failures: {failures}")
                self.stats['messages_failed'] += 1
            else:
                self.stats['messages_delivered'] += 1
                
            # Update latency stats
            latency = (datetime.now() - start_time).total_seconds()
            self._update_latency(latency)
            
        except Exception as e:
            logger.error(f"Error routing message: {e}")
            self.dead_letter_queue.append(message)
            self.stats['messages_failed'] += 1
            
    async def _execute_callback(self, callback: Callable, message: AgentMessage):
        """Execute callback with error handling"""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(message)
            else:
                callback(message)
        except Exception as e:
            logger.error(f"Callback error: {e}")
            raise
            
    def _update_latency(self, latency: float):
        """Update average latency statistics"""
        alpha = 0.1  # Exponential moving average factor
        self.stats['average_latency'] = (
            alpha * latency + 
            (1 - alpha) * self.stats['average_latency']
        )
        
    def get_stats(self) -> Dict[str, Any]:
        """Get message bus statistics"""
        return {
            **self.stats,
            'queue_size': self.message_queue.qsize(),
            'dead_letter_count': len(self.dead_letter_queue),
            'subscriber_count': sum(len(subs) for subs in self.subscribers.values()),
            'history_size': len(self.message_history)
        }
        
    async def replay_messages(self, start_time: datetime, end_time: datetime, 
                            message_type: Optional[str] = None):
        """
        Replay messages from history.
        
        Useful for debugging and recovery scenarios.
        """
        filtered = [
            msg for msg in self.message_history
            if start_time <= msg.timestamp <= end_time
            and (message_type is None or msg.message_type == message_type)
        ]
        
        logger.info(f"Replaying {len(filtered)} messages")
        
        for message in filtered:
            await self._route_message(message)
            
    def get_dead_letters(self) -> List[AgentMessage]:
        """Get messages that failed to deliver"""
        return self.dead_letter_queue.copy()
        
    def clear_dead_letters(self):
        """Clear dead letter queue"""
        self.dead_letter_queue.clear()


class AgentCommunicator:
    """
    High-level interface for agents to communicate via message bus.
    """
    
    def __init__(self, agent_id: str, message_bus: MessageBus):
        self.agent_id = agent_id
        self.message_bus = message_bus
        self.pending_requests: Dict[str, asyncio.Future] = {}
        
    async def send(self, receiver: str, message_type: str, payload: Dict[str, Any],
                  priority: MessagePriority = MessagePriority.NORMAL) -> bool:
        """Send message to another agent"""
        message = AgentMessage(
            sender=self.agent_id,
            receiver=receiver,
            message_type=message_type,
            payload=payload,
            priority=priority
        )
        return await self.message_bus.publish(message)
        
    async def broadcast(self, message_type: str, payload: Dict[str, Any],
                       priority: MessagePriority = MessagePriority.NORMAL) -> bool:
        """Broadcast message to all agents"""
        return await self.send("*", message_type, payload, priority)
        
    async def request(self, receiver: str, message_type: str, payload: Dict[str, Any],
                     timeout: float = 30.0) -> Optional[AgentMessage]:
        """
        Send request and wait for response.
        
        Implements request-response pattern.
        """
        correlation_id = str(uuid.uuid4())
        
        # Create future for response
        future = asyncio.Future()
        self.pending_requests[correlation_id] = future
        
        # Send request
        message = AgentMessage(
            sender=self.agent_id,
            receiver=receiver,
            message_type=message_type,
            payload=payload,
            correlation_id=correlation_id,
            reply_to=self.agent_id
        )
        
        await self.message_bus.publish(message)
        
        try:
            # Wait for response
            response = await asyncio.wait_for(future, timeout=timeout)
            return response
        except asyncio.TimeoutError:
            logger.error(f"Request timeout: {correlation_id}")
            return None
        finally:
            # Clean up
            self.pending_requests.pop(correlation_id, None)
            
    async def respond(self, original_message: AgentMessage, payload: Dict[str, Any]):
        """Send response to request"""
        if not original_message.reply_to:
            logger.warning("Cannot respond to message without reply_to")
            return
            
        response = AgentMessage(
            sender=self.agent_id,
            receiver=original_message.reply_to,
            message_type=f"{original_message.message_type}.response",
            payload=payload,
            correlation_id=original_message.correlation_id
        )
        
        await self.message_bus.publish(response)
        
    def handle_response(self, message: AgentMessage):
        """Handle response to pending request"""
        if message.correlation_id in self.pending_requests:
            future = self.pending_requests[message.correlation_id]
            if not future.done():
                future.set_result(message)
                
    async def subscribe(self, message_type: str, callback: Callable):
        """Subscribe to message type"""
        await self.message_bus.subscribe(message_type, callback, self.agent_id)
        
    async def subscribe_all(self, callback: Callable):
        """Subscribe to all messages"""
        await self.message_bus.subscribe("*", callback, self.agent_id)


# Example usage
if __name__ == "__main__":
    async def example():
        # Create message bus
        bus = MessageBus()
        
        # Start bus
        bus_task = asyncio.create_task(bus.start())
        
        # Create communicators for two agents
        agent1 = AgentCommunicator("agent-1", bus)
        agent2 = AgentCommunicator("agent-2", bus)
        
        # Agent 2 subscribes to messages
        async def handle_message(message: AgentMessage):
            print(f"Agent 2 received: {message.message_type} - {message.payload}")
            
        await agent2.subscribe(MessageType.TASK_REQUEST.value, handle_message)
        
        # Agent 1 sends message
        await agent1.send(
            "agent-2",
            MessageType.TASK_REQUEST.value,
            {"task": "process_manuscript", "manuscript_id": 123}
        )
        
        # Wait a bit for processing
        await asyncio.sleep(2)
        
        # Stop bus
        await bus.stop()
        
        # Print stats
        print(f"Stats: {bus.get_stats()}")
        
    asyncio.run(example())
