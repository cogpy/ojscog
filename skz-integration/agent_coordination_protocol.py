"""
OJSCog Agent Coordination Protocol
Inter-Agent Communication and Workflow Orchestration

This module implements the communication protocol for coordinating the 7 autonomous
agents in the OJSCog system. It provides message passing, state synchronization,
and workflow orchestration capabilities.

Author: OJSCog Team
Date: 2025-11-15
Version: 1.0
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of inter-agent messages"""
    REQUEST = "REQUEST"
    RESPONSE = "RESPONSE"
    NOTIFY = "NOTIFY"
    QUERY = "QUERY"
    COMMAND = "COMMAND"


class MessagePriority(Enum):
    """Message priority levels"""
    CRITICAL = 1
    HIGH = 3
    NORMAL = 5
    LOW = 7
    BACKGROUND = 10


class MessageStatus(Enum):
    """Message processing status"""
    PENDING = "pending"
    SENT = "sent"
    RECEIVED = "received"
    PROCESSED = "processed"
    FAILED = "failed"


@dataclass
class AgentMessage:
    """
    Standard message format for inter-agent communication.
    """
    message_id: str
    message_type: MessageType
    sender_agent_id: str
    receiver_agent_id: str
    timestamp: str
    priority: int
    payload: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None
    status: MessageStatus = MessageStatus.PENDING
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            'message_id': self.message_id,
            'message_type': self.message_type.value,
            'sender_agent_id': self.sender_agent_id,
            'receiver_agent_id': self.receiver_agent_id,
            'timestamp': self.timestamp,
            'priority': self.priority,
            'payload': self.payload,
            'metadata': self.metadata,
            'correlation_id': self.correlation_id,
            'status': self.status.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentMessage':
        """Create message from dictionary."""
        return cls(
            message_id=data['message_id'],
            message_type=MessageType(data['message_type']),
            sender_agent_id=data['sender_agent_id'],
            receiver_agent_id=data['receiver_agent_id'],
            timestamp=data['timestamp'],
            priority=data['priority'],
            payload=data['payload'],
            metadata=data.get('metadata', {}),
            correlation_id=data.get('correlation_id'),
            status=MessageStatus(data.get('status', 'pending'))
        )


class AgentCoordinationProtocol:
    """
    Main coordination protocol for inter-agent communication.
    
    Provides:
    - Message routing and delivery
    - Request-response pattern support
    - Publish-subscribe for notifications
    - State synchronization
    - Error handling and retries
    """
    
    def __init__(self):
        """Initialize the coordination protocol."""
        self.message_queue = asyncio.PriorityQueue()
        self.agent_mailboxes: Dict[str, asyncio.Queue] = defaultdict(asyncio.Queue)
        self.message_handlers: Dict[str, Dict[MessageType, Callable]] = defaultdict(dict)
        self.pending_responses: Dict[str, asyncio.Future] = {}
        self.agent_states: Dict[str, Dict[str, Any]] = {}
        self.subscribers: Dict[str, List[str]] = defaultdict(list)
        
        logger.info("AgentCoordinationProtocol initialized")
    
    def register_agent(self, agent_id: str, handlers: Dict[MessageType, Callable]):
        """
        Register an agent with the coordination protocol.
        
        Args:
            agent_id: Unique agent identifier
            handlers: Dictionary mapping message types to handler functions
        """
        self.message_handlers[agent_id] = handlers
        self.agent_states[agent_id] = {
            'status': 'active',
            'last_seen': datetime.utcnow().isoformat()
        }
        logger.info(f"Agent {agent_id} registered with {len(handlers)} handlers")
    
    def unregister_agent(self, agent_id: str):
        """
        Unregister an agent from the coordination protocol.
        
        Args:
            agent_id: Agent identifier to unregister
        """
        if agent_id in self.message_handlers:
            del self.message_handlers[agent_id]
        if agent_id in self.agent_states:
            del self.agent_states[agent_id]
        logger.info(f"Agent {agent_id} unregistered")
    
    async def send_message(self, message: AgentMessage) -> str:
        """
        Send a message to another agent.
        
        Args:
            message: AgentMessage to send
            
        Returns:
            Message ID
        """
        # Update message status
        message.status = MessageStatus.SENT
        
        # Add to priority queue (lower priority number = higher priority)
        await self.message_queue.put((message.priority, message))
        
        # Add to receiver's mailbox
        await self.agent_mailboxes[message.receiver_agent_id].put(message)
        
        logger.info(f"Message {message.message_id} sent from {message.sender_agent_id} "
                   f"to {message.receiver_agent_id}")
        
        return message.message_id
    
    async def send_request(self, sender_id: str, receiver_id: str, 
                          action: str, parameters: Dict[str, Any],
                          priority: int = MessagePriority.NORMAL.value,
                          timeout: float = 30.0) -> Dict[str, Any]:
        """
        Send a request and wait for response.
        
        Args:
            sender_id: Sender agent ID
            receiver_id: Receiver agent ID
            action: Action to request
            parameters: Request parameters
            priority: Message priority
            timeout: Response timeout in seconds
            
        Returns:
            Response payload
        """
        message_id = str(uuid.uuid4())
        correlation_id = str(uuid.uuid4())
        
        # Create future for response
        response_future = asyncio.Future()
        self.pending_responses[correlation_id] = response_future
        
        # Create request message
        message = AgentMessage(
            message_id=message_id,
            message_type=MessageType.REQUEST,
            sender_agent_id=sender_id,
            receiver_agent_id=receiver_id,
            timestamp=datetime.utcnow().isoformat(),
            priority=priority,
            payload={
                'action': action,
                'parameters': parameters
            },
            correlation_id=correlation_id
        )
        
        # Send message
        await self.send_message(message)
        
        # Wait for response with timeout
        try:
            response = await asyncio.wait_for(response_future, timeout=timeout)
            return response
        except asyncio.TimeoutError:
            logger.error(f"Request {message_id} timed out after {timeout}s")
            del self.pending_responses[correlation_id]
            raise TimeoutError(f"Request to {receiver_id} timed out")
    
    async def send_response(self, request_message: AgentMessage, 
                           response_data: Dict[str, Any]):
        """
        Send a response to a request.
        
        Args:
            request_message: Original request message
            response_data: Response data
        """
        response_message = AgentMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.RESPONSE,
            sender_agent_id=request_message.receiver_agent_id,
            receiver_agent_id=request_message.sender_agent_id,
            timestamp=datetime.utcnow().isoformat(),
            priority=request_message.priority,
            payload=response_data,
            correlation_id=request_message.correlation_id
        )
        
        await self.send_message(response_message)
        
        # Resolve pending future if exists
        if request_message.correlation_id in self.pending_responses:
            future = self.pending_responses[request_message.correlation_id]
            if not future.done():
                future.set_result(response_data)
            del self.pending_responses[request_message.correlation_id]
    
    async def notify(self, sender_id: str, event_type: str, 
                    event_data: Dict[str, Any],
                    priority: int = MessagePriority.NORMAL.value):
        """
        Broadcast a notification to all subscribers.
        
        Args:
            sender_id: Sender agent ID
            event_type: Type of event
            event_data: Event data
            priority: Message priority
        """
        subscribers = self.subscribers.get(event_type, [])
        
        for receiver_id in subscribers:
            message = AgentMessage(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.NOTIFY,
                sender_agent_id=sender_id,
                receiver_agent_id=receiver_id,
                timestamp=datetime.utcnow().isoformat(),
                priority=priority,
                payload={
                    'event_type': event_type,
                    'event_data': event_data
                }
            )
            await self.send_message(message)
        
        logger.info(f"Notification {event_type} sent to {len(subscribers)} subscribers")
    
    def subscribe(self, agent_id: str, event_type: str):
        """
        Subscribe an agent to event notifications.
        
        Args:
            agent_id: Agent ID to subscribe
            event_type: Event type to subscribe to
        """
        if agent_id not in self.subscribers[event_type]:
            self.subscribers[event_type].append(agent_id)
            logger.info(f"Agent {agent_id} subscribed to {event_type}")
    
    def unsubscribe(self, agent_id: str, event_type: str):
        """
        Unsubscribe an agent from event notifications.
        
        Args:
            agent_id: Agent ID to unsubscribe
            event_type: Event type to unsubscribe from
        """
        if agent_id in self.subscribers[event_type]:
            self.subscribers[event_type].remove(agent_id)
            logger.info(f"Agent {agent_id} unsubscribed from {event_type}")
    
    async def receive_messages(self, agent_id: str) -> AgentMessage:
        """
        Receive next message for an agent.
        
        Args:
            agent_id: Agent ID to receive messages for
            
        Returns:
            Next message in mailbox
        """
        message = await self.agent_mailboxes[agent_id].get()
        message.status = MessageStatus.RECEIVED
        
        # Update agent last seen
        if agent_id in self.agent_states:
            self.agent_states[agent_id]['last_seen'] = datetime.utcnow().isoformat()
        
        return message
    
    async def process_message(self, agent_id: str, message: AgentMessage):
        """
        Process a received message.
        
        Args:
            agent_id: Agent ID processing the message
            message: Message to process
        """
        try:
            # Get handler for message type
            if agent_id not in self.message_handlers:
                logger.error(f"No handlers registered for agent {agent_id}")
                return
            
            handler = self.message_handlers[agent_id].get(message.message_type)
            if handler is None:
                logger.warning(f"No handler for {message.message_type} in agent {agent_id}")
                return
            
            # Execute handler
            if asyncio.iscoroutinefunction(handler):
                result = await handler(message)
            else:
                result = handler(message)
            
            # Send response if this was a request
            if message.message_type == MessageType.REQUEST:
                await self.send_response(message, result)
            
            message.status = MessageStatus.PROCESSED
            logger.info(f"Message {message.message_id} processed by {agent_id}")
        
        except Exception as e:
            logger.error(f"Error processing message {message.message_id}: {e}")
            message.status = MessageStatus.FAILED
            
            # Send error response if this was a request
            if message.message_type == MessageType.REQUEST:
                await self.send_response(message, {
                    'error': str(e),
                    'status': 'failed'
                })
    
    async def agent_message_loop(self, agent_id: str):
        """
        Main message processing loop for an agent.
        
        Args:
            agent_id: Agent ID to run loop for
        """
        logger.info(f"Starting message loop for agent {agent_id}")
        
        while True:
            try:
                # Receive next message
                message = await self.receive_messages(agent_id)
                
                # Process message
                await self.process_message(agent_id, message)
            
            except asyncio.CancelledError:
                logger.info(f"Message loop cancelled for agent {agent_id}")
                break
            except Exception as e:
                logger.error(f"Error in message loop for {agent_id}: {e}")
                await asyncio.sleep(1)  # Brief pause before retrying
    
    def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """
        Get current status of an agent.
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Agent status dictionary
        """
        return self.agent_states.get(agent_id, {'status': 'unknown'})
    
    def get_all_agent_statuses(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status of all registered agents.
        
        Returns:
            Dictionary mapping agent IDs to status
        """
        return self.agent_states.copy()


class WorkflowOrchestrator:
    """
    Orchestrates workflows across multiple agents.
    
    Implements workflow patterns:
    - Pipeline: Sequential agent execution
    - Scatter-Gather: Parallel execution with aggregation
    - Saga: Distributed transaction with compensation
    - Circuit Breaker: Fault tolerance
    """
    
    def __init__(self, protocol: AgentCoordinationProtocol):
        """
        Initialize workflow orchestrator.
        
        Args:
            protocol: Agent coordination protocol instance
        """
        self.protocol = protocol
        self.workflows: Dict[str, Dict[str, Any]] = {}
        logger.info("WorkflowOrchestrator initialized")
    
    async def execute_pipeline(self, workflow_id: str, 
                               agents: List[str],
                               initial_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a pipeline workflow (sequential agent execution).
        
        Args:
            workflow_id: Unique workflow identifier
            agents: List of agent IDs in execution order
            initial_data: Initial workflow data
            
        Returns:
            Final workflow result
        """
        logger.info(f"Executing pipeline workflow {workflow_id} with {len(agents)} agents")
        
        current_data = initial_data
        
        for i, agent_id in enumerate(agents):
            logger.info(f"Pipeline step {i+1}/{len(agents)}: {agent_id}")
            
            # Send request to agent
            result = await self.protocol.send_request(
                sender_id='orchestrator',
                receiver_id=agent_id,
                action='process_workflow_step',
                parameters={
                    'workflow_id': workflow_id,
                    'step': i + 1,
                    'data': current_data
                }
            )
            
            # Update data for next step
            current_data = result.get('output', current_data)
        
        logger.info(f"Pipeline workflow {workflow_id} completed")
        return current_data
    
    async def execute_scatter_gather(self, workflow_id: str,
                                    agents: List[str],
                                    data: Dict[str, Any],
                                    aggregator: Callable) -> Dict[str, Any]:
        """
        Execute scatter-gather workflow (parallel execution with aggregation).
        
        Args:
            workflow_id: Unique workflow identifier
            agents: List of agent IDs to execute in parallel
            data: Data to send to all agents
            aggregator: Function to aggregate results
            
        Returns:
            Aggregated result
        """
        logger.info(f"Executing scatter-gather workflow {workflow_id} with {len(agents)} agents")
        
        # Scatter: Send requests to all agents in parallel
        tasks = []
        for agent_id in agents:
            task = self.protocol.send_request(
                sender_id='orchestrator',
                receiver_id=agent_id,
                action='process_parallel_task',
                parameters={
                    'workflow_id': workflow_id,
                    'data': data
                }
            )
            tasks.append(task)
        
        # Wait for all responses
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out errors
        valid_results = [r for r in results if not isinstance(r, Exception)]
        
        # Gather: Aggregate results
        aggregated = aggregator(valid_results)
        
        logger.info(f"Scatter-gather workflow {workflow_id} completed with "
                   f"{len(valid_results)}/{len(agents)} successful results")
        
        return aggregated
    
    async def execute_saga(self, workflow_id: str,
                          steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute saga workflow (distributed transaction with compensation).
        
        Args:
            workflow_id: Unique workflow identifier
            steps: List of step dictionaries with 'agent', 'action', 'compensation'
            
        Returns:
            Workflow result or compensation result
        """
        logger.info(f"Executing saga workflow {workflow_id} with {len(steps)} steps")
        
        completed_steps = []
        
        try:
            # Execute each step
            for i, step in enumerate(steps):
                logger.info(f"Saga step {i+1}/{len(steps)}: {step['agent']}")
                
                result = await self.protocol.send_request(
                    sender_id='orchestrator',
                    receiver_id=step['agent'],
                    action=step['action'],
                    parameters=step.get('parameters', {})
                )
                
                completed_steps.append({
                    'step': i,
                    'agent': step['agent'],
                    'result': result,
                    'compensation': step.get('compensation')
                })
            
            logger.info(f"Saga workflow {workflow_id} completed successfully")
            return {'status': 'success', 'results': completed_steps}
        
        except Exception as e:
            logger.error(f"Saga workflow {workflow_id} failed at step {len(completed_steps)}: {e}")
            
            # Execute compensation in reverse order
            logger.info(f"Executing compensation for {len(completed_steps)} completed steps")
            
            for step_data in reversed(completed_steps):
                if step_data['compensation']:
                    try:
                        await self.protocol.send_request(
                            sender_id='orchestrator',
                            receiver_id=step_data['agent'],
                            action=step_data['compensation'],
                            parameters={'original_result': step_data['result']}
                        )
                    except Exception as comp_error:
                        logger.error(f"Compensation failed for step {step_data['step']}: {comp_error}")
            
            return {'status': 'compensated', 'error': str(e)}


# Example usage
if __name__ == "__main__":
    async def main():
        # Initialize protocol
        protocol = AgentCoordinationProtocol()
        
        # Register agents with handlers
        async def handle_request(message: AgentMessage) -> Dict[str, Any]:
            logger.info(f"Agent handling request: {message.payload}")
            return {'status': 'success', 'output': 'processed'}
        
        protocol.register_agent('agent_001', {
            MessageType.REQUEST: handle_request
        })
        
        protocol.register_agent('agent_002', {
            MessageType.REQUEST: handle_request
        })
        
        # Subscribe to events
        protocol.subscribe('agent_002', 'manuscript_submitted')
        
        # Send a request
        response = await protocol.send_request(
            sender_id='agent_001',
            receiver_id='agent_002',
            action='analyze_manuscript',
            parameters={'manuscript_id': '123'}
        )
        
        logger.info(f"Response: {response}")
        
        # Send a notification
        await protocol.notify(
            sender_id='agent_001',
            event_type='manuscript_submitted',
            event_data={'manuscript_id': '123'}
        )
        
        # Initialize orchestrator
        orchestrator = WorkflowOrchestrator(protocol)
        
        # Execute pipeline
        result = await orchestrator.execute_pipeline(
            workflow_id='wf_001',
            agents=['agent_001', 'agent_002'],
            initial_data={'manuscript_id': '123'}
        )
        
        logger.info(f"Pipeline result: {result}")
    
    # Run example
    asyncio.run(main())
