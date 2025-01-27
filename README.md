# Advanced Python Interview Questions and Answers (8+ Years Experience)

## Core Python Concepts

### 1. Advanced Memory Management
Q: Explain Python's memory management strategies and how you would handle memory leaks in a production environment?

A: Python's memory management is sophisticated but requires careful attention:
- Memory is managed through private heaps containing Python objects and data structures
- Key strategies:
  1. Reference counting for basic memory management
  2. Generational garbage collection for cyclic references
  3. Memory pooling for small objects
  
For handling memory leaks:
```python
import gc
import tracemalloc

# Start trace
tracemalloc.start()

# Your code here
def potential_memory_leak():
    large_dict = {}
    # ... operations

# Get memory snapshot
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')

# Analysis
print("[ Top 10 memory users ]")
for stat in top_stats[:10]:
    print(stat)
```

### 2. Advanced Concurrency
Q: Compare and contrast Python's different concurrency models and their use cases.

A: Python offers multiple concurrency models:

1. Threading (concurrent.futures.ThreadPoolExecutor):
```python
from concurrent.futures import ThreadPoolExecutor
import requests

def fetch_url(url):
    return requests.get(url).text

urls = ['http://example1.com', 'http://example2.com']
with ThreadPoolExecutor(max_workers=3) as executor:
    results = executor.map(fetch_url, urls)
```

2. Multiprocessing:
```python
from multiprocessing import Pool

def cpu_intensive_task(data):
    return sum(x * x for x in range(data))

if __name__ == '__main__':
    with Pool(processes=4) as pool:
        results = pool.map(cpu_intensive_task, range(10))
```

3. Asynchronous I/O:
```python
import asyncio

async def async_task(name, delay):
    await asyncio.sleep(delay)
    return f'{name} completed'

async def main():
    tasks = [
        async_task('Task1', 2),
        async_task('Task2', 1)
    ]
    results = await asyncio.gather(*tasks)

asyncio.run(main())
```

### 3. Design Patterns and Architecture
Q: Explain how you would implement a plugin system in Python that dynamically loads modules.

A: Here's an implementation of a plugin system:

```python
import importlib
import inspect
from abc import ABC, abstractmethod

class PluginInterface(ABC):
    @abstractmethod
    def process(self, data):
        pass

class PluginLoader:
    def __init__(self, plugin_dir):
        self.plugin_dir = plugin_dir
        self.plugins = {}

    def load_plugin(self, plugin_name):
        module = importlib.import_module(f"{self.plugin_dir}.{plugin_name}")
        for name, cls in inspect.getmembers(module, inspect.isclass):
            if issubclass(cls, PluginInterface) and cls != PluginInterface:
                self.plugins[plugin_name] = cls()

    def execute_plugin(self, plugin_name, data):
        return self.plugins[plugin_name].process(data)
```

### 4. Advanced Metaprogramming
Q: How would you implement a declarative API using metaclasses and descriptors?

A: Here's an example of a declarative API for data validation:

```python
class Validator:
    def __init__(self, validation_func, error_msg):
        self.validation_func = validation_func
        self.error_msg = error_msg

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__.get(self.name)

    def __set__(self, instance, value):
        if not self.validation_func(value):
            raise ValueError(f'{self.name}: {self.error_msg}')
        instance.__dict__[self.name] = value

    def __set_name__(self, owner, name):
        self.name = name

class ModelMeta(type):
    def __new__(cls, name, bases, namespace):
        for key, value in namespace.items():
            if isinstance(value, Validator):
                value.__set_name__(cls, key)
        return super().__new__(cls, name, bases, namespace)

class Model(metaclass=ModelMeta):
    pass

class User(Model):
    name = Validator(lambda x: isinstance(x, str) and x.strip(),
                    'Name must be a non-empty string')
    age = Validator(lambda x: isinstance(x, int) and 0 <= x <= 150,
                   'Age must be between 0 and 150')
```

### 5. Performance Optimization
Q: How would you optimize a Python application that processes large datasets?

A: Here's a comprehensive approach to optimization:

1. Data Processing Optimization:
```python
# Instead of this
def process_large_file(filename):
    with open(filename) as f:
        data = f.readlines()  # Loads entire file into memory
    return [process_line(line) for line in data]

# Use this
def process_large_file(filename):
    with open(filename) as f:
        for line in f:  # Generator-based approach
            yield process_line(line)
```

2. Numpy for Numerical Operations:
```python
import numpy as np

# Instead of this
def calculate_stats(data):
    mean = sum(data) / len(data)
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    return mean, variance

# Use this
def calculate_stats(data):
    data_array = np.array(data)
    return np.mean(data_array), np.var(data_array)
```

### 6. Advanced Testing Strategies
Q: How would you implement property-based testing in Python and what are its advantages?

A: Property-based testing helps discover edge cases by generating random test inputs:

```python
from hypothesis import given, strategies as st

class NumberOperations:
    def add_and_multiply(self, x, y):
        return (x + y, x * y)

# Property-based test
@given(st.integers(), st.integers())
def test_add_and_multiply(x, y):
    ops = NumberOperations()
    sum_result, product_result = ops.add_and_multiply(x, y)
    
    # Properties that should always hold
    assert sum_result == x + y
    assert product_result == x * y
    assert isinstance(sum_result, int)
    assert isinstance(product_result, int)
```

### 7. Security and Authentication
Q: Explain how you would implement secure password storage and API authentication in Python.

A: Here's a secure implementation using modern best practices:

```python
import hashlib
import hmac
import secrets
from base64 import b64encode
from datetime import datetime, timedelta
import jwt

class SecurityManager:
    def __init__(self, secret_key):
        self.secret_key = secret_key

    def hash_password(self, password: str) -> tuple:
        salt = secrets.token_bytes(32)
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt,
            100000  # Number of iterations
        )
        return b64encode(salt).decode('utf-8'), b64encode(password_hash).decode('utf-8')

    def verify_password(self, password: str, stored_salt: str, stored_hash: str) -> bool:
        salt = b64encode(stored_salt.encode('utf-8'))
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt,
            100000
        )
        return hmac.compare_digest(
            b64encode(password_hash).decode('utf-8'),
            stored_hash
        )

    def generate_jwt(self, user_id: int) -> str:
        payload = {
            'user_id': user_id,
            'exp': datetime.utcnow() + timedelta(days=1),
            'iat': datetime.utcnow()
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')

    def verify_jwt(self, token: str) -> dict:
        try:
            return jwt.decode(token, self.secret_key, algorithms=['HS256'])
        except jwt.InvalidTokenError:
            return None
```

### 8. Advanced Database Patterns
Q: How would you implement a pattern for handling complex database transactions with retry logic and connection pooling?

A: Here's an implementation using SQLAlchemy with connection pooling and retry logic:

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import OperationalError
from contextlib import contextmanager
import time
import logging

class DatabaseManager:
    def __init__(self, connection_string, pool_size=5, max_retries=3):
        self.engine = create_engine(
            connection_string,
            pool_size=pool_size,
            pool_timeout=30,
            pool_recycle=3600
        )
        self.Session = sessionmaker(bind=self.engine)
        self.max_retries = max_retries

    @contextmanager
    def session_scope(self):
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise
        finally:
            session.close()

    def execute_with_retry(self, operation):
        retries = 0
        while retries < self.max_retries:
            try:
                with self.session_scope() as session:
                    result = operation(session)
                    return result
            except OperationalError as e:
                retries += 1
                if retries == self.max_retries:
                    raise
                time.sleep(2 ** retries)  # Exponential backoff
                logging.warning(f"Retry attempt {retries} for database operation")

# Usage example
db = DatabaseManager("postgresql://user:pass@localhost/db")

def update_user_status(user_id: int, new_status: str):
    def operation(session):
        user = session.query(User).filter_by(id=user_id).first()
        user.status = new_status
        return user

    return db.execute_with_retry(operation)
```

### 9. System Design and Scalability
Q: Design a distributed task queue system with Python. How would you handle task prioritization and failure recovery?

A: Here's an implementation of a distributed task queue:

```python
import redis
import json
import uuid
from datetime import datetime
from typing import Any, Callable, Optional
from dataclasses import dataclass
import pickle

@dataclass
class Task:
    id: str
    function: str
    args: tuple
    kwargs: dict
    priority: int
    retry_count: int = 0
    max_retries: int = 3
    status: str = 'pending'
    result: Any = None

class DistributedTaskQueue:
    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url)
        self.task_queues = {
            'high': 'task_queue:high',
            'medium': 'task_queue:medium',
            'low': 'task_queue:low'
        }
        self.result_key = 'task_results'
        self.dead_letter_queue = 'dead_letter_queue'

    def enqueue(self, func: Callable, *args, priority: str = 'medium', **kwargs) -> str:
        task = Task(
            id=str(uuid.uuid4()),
            function=pickle.dumps(func),
            args=args,
            kwargs=kwargs,
            priority=self.task_queues[priority]
        )
        
        self.redis.lpush(
            self.task_queues[priority],
            json.dumps(task.__dict__)
        )
        return task.id

    def process_task(self, task_data: dict) -> None:
        task = Task(**json.loads(task_data))
        try:
            func = pickle.loads(task.function)
            result = func(*task.args, **task.kwargs)
            task.status = 'completed'
            task.result = result
        except Exception as e:
            task.retry_count += 1
            if task.retry_count >= task.max_retries:
                task.status = 'failed'
                self.redis.lpush(
                    self.dead_letter_queue,
                    json.dumps(task.__dict__)
                )
            else:
                # Requeue with exponential backoff
                self.redis.lpush(
                    task.priority,
                    json.dumps(task.__dict__)
                )
        finally:
            self.redis.hset(
                self.result_key,
                task.id,
                json.dumps(task.__dict__)
            )

    def worker(self, priorities: list = ['high', 'medium', 'low']):
        while True:
            for priority in priorities:
                task_data = self.redis.brpop(
                    self.task_queues[priority],
                    timeout=1
                )
                if task_data:
                    self.process_task(task_data[1])

    def get_result(self, task_id: str) -> Optional[dict]:
        result = self.redis.hget(self.result_key, task_id)
        return json.loads(result) if result else None
```

### 10. Microservices Architecture
Q: How would you implement service discovery and circuit breaking patterns in a Python microservices architecture?

A: Here's an implementation combining service discovery with circuit breaking:

```python
from typing import Dict, List, Optional
import requests
from datetime import datetime, timedelta
import threading
import time
import logging

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, reset_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failures = 0
        self.last_failure_time = None
        self.state = 'closed'
        self._lock = threading.Lock()

    def can_execute(self) -> bool:
        with self._lock:
            if self.state == 'open':
                if (datetime.now() - self.last_failure_time).seconds >= self.reset_timeout:
                    self.state = 'half-open'
                    return True
                return False
            return True

    def record_failure(self):
        with self._lock:
            self.failures += 1
            self.last_failure_time = datetime.now()
            if self.failures >= self.failure_threshold:
                self.state = 'open'

    def record_success(self):
        with self._lock:
            self.failures = 0
            self.state = 'closed'

class ServiceRegistry:
    def __init__(self, consul_url: str):
        self.consul_url = consul_url
        self.services: Dict[str, List[str]] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()

    def register_service(self, service_name: str, instance_url: str):
        with self._lock:
            if service_name not in self.services:
                self.services[service_name] = []
            self.services[service_name].append(instance_url)
            self.circuit_breakers[instance_url] = CircuitBreaker()

    def get_service_instance(self, service_name: str) -> Optional[str]:
        with self._lock:
            instances = self.services.get(service_name, [])
            for instance in instances:
                circuit_breaker = self.circuit_breakers[instance]
                if circuit_breaker.can_execute():
                    return instance
            return None

class ServiceClient:
    def __init__(self, service_registry: ServiceRegistry):
        self.service_registry = service_registry

    def call_service(self, service_name: str, endpoint: str, method: str = 'GET', **kwargs):
        instance_url = self.service_registry.get_service_instance(service_name)
        if not instance_url:
            raise Exception(f"No available instances for service {service_name}")

        circuit_breaker = self.service_registry.circuit_breakers[instance_url]
        
        try:
            response = requests.request(
                method,
                f"{instance_url}/{endpoint.lstrip('/')}",
                **kwargs
            )
            response.raise_for_status()
            circuit_breaker.record_success()
            return response.json()
        except requests.exceptions.RequestException as e:
            circuit_breaker.record_failure()
            logging.error(f"Service call failed: {str(e)}")
            raise

# Usage example
registry = ServiceRegistry("http://consul:8500")
registry.register_service("user-service", "http://user-service:8080")
registry.register_service("user-service", "http://user-service:8081")

client = ServiceClient(registry)
try:
    result = client.call_service("user-service", "/api/users", method="GET")
except Exception as e:
    print(f"Service call failed: {str(e)}")
```
### 11. Cloud Development (AWS)
Q: How would you implement a serverless ETL pipeline using Python and AWS services?

A: Here's an implementation using AWS Lambda and S3:

```python
import boto3
import pandas as pd
import json
from io import StringIO
from typing import Dict, List

class ETLPipeline:
    def __init__(self):
        self.s3 = boto3.client('s3')
        self.sqs = boto3.client('sqs')
        self.dynamodb = boto3.resource('dynamodb')

    def extract_from_s3(self, bucket: str, key: str) -> pd.DataFrame:
        """Extract data from S3 bucket"""
        try:
            obj = self.s3.get_object(Bucket=bucket, Key=key)
            data = pd.read_csv(obj['Body'])
            return data
        except Exception as e:
            print(f"Error extracting from S3: {str(e)}")
            raise

    def transform_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply transformations to the data"""
        # Example transformations
        transformed = data.copy()
        transformed['timestamp'] = pd.to_datetime(transformed['timestamp'])
        transformed['year'] = transformed['timestamp'].dt.year
        transformed['month'] = transformed['timestamp'].dt.month
        
        # Aggregate data
        aggregated = transformed.groupby(['year', 'month']).agg({
            'value': ['mean', 'sum', 'count']
        }).reset_index()
        
        return aggregated

    def load_to_dynamodb(self, data: pd.DataFrame, table_name: str):
        """Load transformed data to DynamoDB"""
        table = self.dynamodb.Table(table_name)
        
        with table.batch_writer() as batch:
            for _, row in data.iterrows():
                batch.put_item(Item=row.to_dict())

def lambda_handler(event, context):
    """AWS Lambda handler function"""
    pipeline = ETLPipeline()
    
    # Get S3 event details
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']
    
    try:
        # Execute ETL pipeline
        raw_data = pipeline.extract_from_s3(bucket, key)
        transformed_data = pipeline.transform_data(raw_data)
        pipeline.load_to_dynamodb(transformed_data, 'processed_data')
        
        return {
            'statusCode': 200,
            'body': json.dumps('ETL pipeline executed successfully')
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error: {str(e)}')
        }
```

### 12. API Design (GraphQL)
Q: Implement a GraphQL API using Python that includes authentication, data loaders, and error handling.

A: Here's an implementation using Graphene:

```python
import graphene
from graphene import relay
from graphql import GraphQLError
from functools import wraps
from promise import Promise
from promise.dataloader import DataLoader
import jwt
from typing import Dict, List

# Authentication decorator
def require_auth(func):
    @wraps(func)
    def wrapped(self, info, **kwargs):
        token = info.context.headers.get('Authorization')
        if not token:
            raise GraphQLError('No auth token provided')
        
        try:
            payload = jwt.decode(token, 'secret', algorithms=['HS256'])
            info.context.user = payload
            return func(self, info, **kwargs)
        except jwt.InvalidTokenError:
            raise GraphQLError('Invalid auth token')
    return wrapped

# Data Loader
class UserLoader(DataLoader):
    def batch_load_fn(self, user_ids):
        # Simulate database batch query
        users = User.objects.filter(id__in=user_ids)
        user_map = {user.id: user for user in users}
        return Promise.resolve([user_map.get(user_id) for user_id in user_ids])

# Types
class UserType(graphene.ObjectType):
    class Meta:
        interfaces = (relay.Node,)
    
    id = graphene.ID(required=True)
    name = graphene.String(required=True)
    email = graphene.String(required=True)
    posts = graphene.List(lambda: PostType)

    def resolve_posts(self, info):
        return Post.objects.filter(author_id=self.id)

class PostType(graphene.ObjectType):
    class Meta:
        interfaces = (relay.Node,)
    
    id = graphene.ID(required=True)
    title = graphene.String(required=True)
    content = graphene.String(required=True)
    author = graphene.Field(UserType)

    def resolve_author(self, info):
        return info.context.loaders.user.load(self.author_id)

# Mutations
class CreatePost(graphene.Mutation):
    class Arguments:
        title = graphene.String(required=True)
        content = graphene.String(required=True)

    post = graphene.Field(PostType)
    
    @require_auth
    def mutate(self, info, title, content):
        user = info.context.user
        post = Post.objects.create(
            title=title,
            content=content,
            author_id=user['id']
        )
        return CreatePost(post=post)

# Query
class Query(graphene.ObjectType):
    node = relay.Node.Field()
    user = graphene.Field(UserType, id=graphene.ID(required=True))
    posts = graphene.List(PostType)

    @require_auth
    def resolve_user(self, info, id):
        return info.context.loaders.user.load(id)

    @require_auth
    def resolve_posts(self, info):
        return Post.objects.all()

# Mutation
class Mutation(graphene.ObjectType):
    create_post = CreatePost.Field()

# Schema
schema = graphene.Schema(query=Query, mutation=Mutation)

# Context Class
class GraphQLContext:
    def __init__(self, request):
        self.request = request
        self.headers = request.headers
        self.user = None
        self.loaders = self.create_loaders()

    def create_loaders(self):
        return type('Loaders', (), {
            'user': UserLoader()
        })

# Error Handling Middleware
class ErrorHandlingMiddleware:
    def resolve(self, next, root, info, **kwargs):
        try:
            return next(root, info, **kwargs)
        except Exception as e:
            if isinstance(e, GraphQLError):
                raise
            return GraphQLError(f'Internal server error: {str(e)}')
```

### 13. Machine Learning Integration
Q: How would you implement a production-ready machine learning pipeline in Python?

A: Here's an implementation of a ML pipeline with model versioning and monitoring:

```python
import mlflow
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from typing import Dict, List, Any
import joblib
import logging
from datetime import datetime

class ModelManager:
    def __init__(self, model_name: str, mlflow_tracking_uri: str):
        self.model_name = model_name
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        self.logger = logging.getLogger(__name__)

    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log metrics to MLflow"""
        with mlflow.start_run():
            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name, value, step=step)

    def save_model(self, model: Pipeline, metrics: Dict[str, float]) -> str:
        """Save model with metrics and parameters"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = f"models/{self.model_name}/{timestamp}"
        
        with mlflow.start_run():
            # Log model parameters
            model_params = self._get_model_params(model)
            mlflow.log_params(model_params)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Save model
            mlflow.sklearn.log_model(model, "model")
            
            # Log model version
            mlflow.set_tag("model_version", timestamp)
            
            return model_path

    def load_model(self, version: str = 'latest') -> Pipeline:
        """Load model by version"""
        if version == 'latest':
            model_uri = f"models:/{self.model_name}/Production"
        else:
            model_uri = f"models:/{self.model_name}/{version}"
            
        return mlflow.sklearn.load_model(model_uri)

    def _get_model_params(self, model: Pipeline) -> Dict[str, Any]:
        """Extract model parameters"""
        params = {}
        for name, step in model.named_steps.items():
            step_params = step.get_params()
            params.update({f"{name}__{k}": v for k, v in step_params.items()})
        return params

class CustomFeatureTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer for feature engineering"""
    def __init__(self, features: List[str]):
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        # Add feature engineering steps here
        return X_copy

class ModelMonitor:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.logger = logging.getLogger(__name__)

    def monitor_predictions(self, inputs: pd.DataFrame, predictions: np.ndarray):
        """Monitor model predictions"""
        metrics = {
            'prediction_mean': float(np.mean(predictions)),
            'prediction_std': float(np.std(predictions)),
            'input_features_null_count': int(inputs.isnull().sum().sum())
        }
        
        self.model_manager.log_metrics(metrics)
        
        # Check for drift
        if self._detect_drift(inputs, predictions):
            self.logger.warning("Data drift detected!")

    def _detect_drift(self, inputs: pd.DataFrame, predictions: np.ndarray) -> bool:
        """Detect data drift in predictions"""
        # Implement drift detection logic
        return False

# Usage Example
def train_and_deploy_model():
    # Initialize model manager
    model_manager = ModelManager(
        model_name="customer_churn",
        mlflow_tracking_uri="sqlite:///mlflow.db"
    )

    # Create pipeline
    pipeline = Pipeline([
        ('feature_engineering', CustomFeatureTransformer(features=['feature1', 'feature2'])),
        ('model', RandomForestClassifier())
    ])

    # Train model
    pipeline.fit(X_train, y_train)

    # Evaluate model
    metrics = {
        'accuracy': accuracy_score(y_test, pipeline.predict(X_test)),
        'precision': precision_score(y_test, pipeline.predict(X_test)),
        'recall': recall_score(y_test, pipeline.predict(X_test))
    }

    # Save model
    model_path = model_manager.save_model(pipeline, metrics)

    # Initialize monitor
    monitor = ModelMonitor(model_manager)

    # Make predictions
    predictions = pipeline.predict(X_test)
    monitor.monitor_predictions(X_test, predictions)

```

### 14. Asynchronous Programming Patterns
Q: Implement a rate-limited asynchronous API client with connection pooling and retry logic.

A: Here's an implementation using aiohttp and async/await:

```python
import aiohttp
import asyncio
from typing import Dict, Optional, Any
import time
import logging
from dataclasses import dataclass
import backoff
from aiohttp import ClientTimeout
from asyncio import Semaphore

@dataclass
class RateLimitConfig:
    requests_per_second: int
    burst_size: Optional[int] = None
    
    def __post_init__(self):
        self.burst_size = self.burst_size or self.requests_per_second * 2

class AsyncAPIClient:
    def __init__(
        self,
        base_url: str,
        rate_limit: RateLimitConfig,
        max_connections: int = 100,
        timeout: int = 30
    ):
        self.base_url = base_url
        self.rate_limit = rate_limit
        self.timeout = ClientTimeout(total=timeout)
        self.semaphore = Semaphore(max_connections)
        self.session: Optional[aiohttp.ClientSession] = None
        self.request_timestamps: List[float] = []
        self.logger = logging.getLogger(__name__)

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=self.timeout,
            raise_for_status=True
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def _wait_for_rate_limit(self):
        """Implement token bucket rate limiting"""
        now = time.time()
        
        # Remove old timestamps
        self.request_timestamps = [
            ts for ts in self.request_timestamps
            if now - ts <= 1.0
        ]
        
        # Check if we're at the rate limit
        if len(self.request_timestamps) >= self.rate_limit.burst_size:
            sleep_time = 1.0 - (now - self.request_timestamps[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        self.request_timestamps.append(now)

    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError),
        max_tries=3
    )
    async def request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Make rate-limited HTTP request with retries"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use 'async with' context manager.")

        await self._wait_for_rate_limit()
        
        async with self.semaphore:
            url = f"{self.base_url}/{endpoint.lstrip('/')}"
            
            try:
                async with self.session.request(method, url, **kwargs) as response:
                    return await response.json()
            except aiohttp.ClientError as e:
                self.logger.error(f"Request failed: {str(e)}")
                raise

    async def get(self, endpoint: str, **kwargs) -> Dict[str, Any]:
        return await self.request('GET', endpoint, **kwargs)

    async def post(self, endpoint: str, **kwargs) -> Dict[str, Any]:
        return await self.request('POST', endpoint, **kwargs)

# Usage Example
async def main():
    rate_limit = RateLimitConfig(requests_per_second=10)
    
    async with AsyncAPIClient(
        base_url="https://api.example.com",
        rate_limit=rate_limit
    ) as client:
        # Make concurrent requests
        tasks = [
            client.get(f"/users/{i}")
            for i in range(100)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for result in results:
            if isinstance(result, Exception):
### 15. Performance Profiling and Optimization
Q: How would you implement a custom profiler to identify bottlenecks in a Python application?

A: Here's an implementation of a custom profiler with detailed metrics:

```python
import time
import cProfile
import pstats
import functools
import threading
from typing import Dict, List, Optional
import logging
from contextlib import contextmanager
import tracemalloc
from dataclasses import dataclass

@dataclass
class ProfileMetrics:
    execution_time: float
    memory_usage: int
    call_count: int
    cpu_time: float

class CustomProfiler:
    def __init__(self):
        self.profiler = cProfile.Profile()
        self.metrics: Dict[str, List[ProfileMetrics]] = {}
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)

    @contextmanager
    def profile_section(self, section_name: str):
        """Context manager for profiling code sections"""
        tracemalloc.start()
        start_time = time.perf_counter()
        start_memory = tracemalloc.get_traced_memory()[0]
        
        try:
            self.profiler.enable()
            yield
        finally:
            self.profiler.disable()
            end_time = time.perf_counter()
            end_memory = tracemalloc.get_traced_memory()[0]
            tracemalloc.stop()
            
            stats = pstats.Stats(self.profiler)
            
            metrics = ProfileMetrics(
                execution_time=end_time - start_time,
                memory_usage=end_memory - start_memory,
                call_count=stats.total_calls,
                cpu_time=stats.total_tt
            )
            
            with self.lock:
                if section_name not in self.metrics:
                    self.metrics[section_name] = []
                self.metrics[section_name].append(metrics)

    def profile_function(self, func):
        """Decorator for profiling functions"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self.profile_section(func.__name__):
                return func(*args, **kwargs)
        return wrapper

    def get_metrics(self, section_name: Optional[str] = None) -> Dict[str, List[ProfileMetrics]]:
        """Get profiling metrics for a section or all sections"""
        with self.lock:
            if section_name:
                return {section_name: self.metrics.get(section_name, [])}
            return self.metrics.copy()

    def analyze_performance(self, section_name: str) -> Dict[str, float]:
        """Analyze performance metrics for a section"""
        metrics = self.metrics.get(section_name, [])
        if not metrics:
            return {}
        
        return {
            'avg_execution_time': sum(m.execution_time for m in metrics) / len(metrics),
            'avg_memory_usage': sum(m.memory_usage for m in metrics) / len(metrics),
            'total_calls': sum(m.call_count for m in metrics),
            'avg_cpu_time': sum(m.cpu_time for m in metrics) / len(metrics)
        }

# Usage example
profiler = CustomProfiler()

@profiler.profile_function
def expensive_operation(n: int):
    return sum(i * i for i in range(n))

def main():
    with profiler.profile_section("main_processing"):
        result = expensive_operation(1000000)
        # More operations...
    
    # Get performance metrics
    metrics = profiler.analyze_performance("expensive_operation")
    print(f"Performance metrics: {metrics}")
```

### 16. Advanced System Design
Q: Design a distributed caching system with eventual consistency and conflict resolution.

A: Here's an implementation of a distributed cache with vector clocks:

```python
import time
from typing import Dict, List, Tuple, Optional, Any
import hashlib
import json
from dataclasses import dataclass
from collections import defaultdict
import threading
import redis
from redis.exceptions import WatchError

@dataclass
class VectorClock:
    node_id: str
    counters: Dict[str, int]

    def increment(self):
        self.counters[self.node_id] = self.counters.get(self.node_id, 0) + 1

    def merge(self, other: 'VectorClock'):
        """Merge two vector clocks"""
        for node, counter in other.counters.items():
            self.counters[node] = max(self.counters.get(node, 0), counter)

    def compare(self, other: 'VectorClock') -> str:
        """Compare vector clocks to determine causality"""
        greater = False
        lesser = False
        
        for node in set(self.counters) | set(other.counters):
            self_count = self.counters.get(node, 0)
            other_count = other.counters.get(node, 0)
            
            if self_count > other_count:
                greater = True
            elif self_count < other_count:
                lesser = True
                
        if greater and not lesser:
            return "greater"
        elif lesser and not greater:
            return "lesser"
        elif not greater and not lesser:
            return "equal"
        else:
            return "concurrent"

@dataclass
class CacheEntry:
    value: Any
    vector_clock: VectorClock
    timestamp: float
    version: str

class DistributedCache:
    def __init__(
        self,
        node_id: str,
        redis_url: str,
        ttl: int = 3600
    ):
        self.node_id = node_id
        self.redis = redis.from_url(redis_url)
        self.ttl = ttl
        self.vector_clock = VectorClock(node_id, {})
        self.local_cache: Dict[str, CacheEntry] = {}
        self.lock = threading.Lock()

    def _generate_version(self, value: Any) -> str:
        """Generate version hash for value"""
        return hashlib.sha256(
            json.dumps(value, sort_keys=True).encode()
        ).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache with vector clock"""
        # Try local cache first
        with self.lock:
            local_entry = self.local_cache.get(key)
            if local_entry:
                return local_entry.value

        # Try Redis
        try:
            with self.redis.pipeline() as pipe:
                while True:
                    try:
                        pipe.watch(f"cache:{key}")
                        entry_data = pipe.get(f"cache:{key}")
                        
                        if not entry_data:
                            return None
                            
                        entry = CacheEntry(**json.loads(entry_data))
                        
                        # Update local cache
                        with self.lock:
                            self.local_cache[key] = entry
                            self.vector_clock.merge(entry.vector_clock)
                        
                        return entry.value
                        
                    except WatchError:
                        continue
        except Exception as e:
            logging.error(f"Error getting from cache: {str(e)}")
            return None

    def set(self, key: str, value: Any):
        """Set value in cache with vector clock"""
        try:
            with self.redis.pipeline() as pipe:
                while True:
                    try:
                        pipe.watch(f"cache:{key}")
                        
                        # Increment vector clock
                        self.vector_clock.increment()
                        
                        entry = CacheEntry(
                            value=value,
                            vector_clock=self.vector_clock,
                            timestamp=time.time(),
                            version=self._generate_version(value)
                        )
                        
                        # Update Redis
                        pipe.multi()
                        pipe.setex(
                            f"cache:{key}",
                            self.ttl,
                            json.dumps(entry.__dict__)
                        )
                        pipe.execute()
                        
                        # Update local cache
                        with self.lock:
                            self.local_cache[key] = entry
                            
                        break
                        
                    except WatchError:
                        continue
                        
        except Exception as e:
            logging.error(f"Error setting cache: {str(e)}")

    def resolve_conflicts(self, key: str, entries: List[CacheEntry]) -> CacheEntry:
        """Resolve conflicts using Last-Writer-Wins with vector clocks"""
        if not entries:
            raise ValueError("No entries to resolve")
            
        # Sort by timestamp and version
        sorted_entries = sorted(
            entries,
            key=lambda e: (e.timestamp, e.version),
            reverse=True
        )
        
        # Find entries with concurrent vector clocks
        latest = sorted_entries[0]
        concurrent_entries = [
            e for e in sorted_entries[1:]
            if e.vector_clock.compare(latest.vector_clock) == "concurrent"
        ]
        
        if not concurrent_entries:
            return latest
            
        # Resolve using version hash
        return max(
            [latest] + concurrent_entries,
            key=lambda e: e.version
        )

# Usage Example
def main():
    cache1 = DistributedCache("node1", "redis://localhost:6379/0")
    cache2 = DistributedCache("node2", "redis://localhost:6379/0")
    
    # Set values from different nodes
    cache1.set("key1", {"data": "value1"})
    cache2.set("key1", {"data": "value2"})
    
    # Get values and resolve conflicts
    value1 = cache1.get("key1")
    value2 = cache2.get("key1")
```

### 17. Real-time Data Processing
Q: Implement a real-time data processing pipeline using Python generators and async streams.

A: Here's an implementation of a streaming data processor:

```python
import asyncio
from typing import AsyncIterator, Dict, List, Any
from dataclasses import dataclass
import json
import aiohttp
import aiokafka
from datetime import datetime
import logging

@dataclass
class DataEvent:
    id: str
    timestamp: datetime
    data: Dict[str, Any]
    source: str

class DataProcessor:
    def __init__(self, batch_size: int = 100):
        self.batch_size = batch_size
        self.logger = logging.getLogger(__name__)
        self.processors: Dict[str, asyncio.Queue] = {}

    async def process_stream(
        self,
        stream: AsyncIterator[DataEvent],
        processor_id: str
    ):
        """Process a stream of data events"""
        buffer = []
        
        async for event in stream:
            buffer.append(event)
            
            if len(buffer) >= self.batch_size:
                await self._process_batch(buffer, processor_id)
                buffer = []
                
        # Process remaining events
        if buffer:
            await self._process_batch(buffer, processor_id)

    async def _process_batch(
        self,
        events: List[DataEvent],
        processor_id: str
    ):
        """Process a batch of events"""
        try:
            # Group events by source
            events_by_source = defaultdict(list)
            for event in events:
                events_by_source[event.source].append(event)
                
            # Process each source's events
            for source, source_events in events_by_source.items():
                processed_data = await self._transform_events(source_events)
                await self._send_to_queue(processed_data, processor_id)
                
        except Exception as e:
            self.logger.error(f"Error processing batch: {str(e)}")
            # Handle error (retry, dead letter queue, etc.)

    async def _transform_events(
        self,
        events: List[DataEvent]
    ) -> Dict[str, Any]:
        """Transform events into processed data"""
        # Implement transformation logic
        return {
            'timestamp': datetime.utcnow(),
            'event_count': len(events),
            'sources': list(set(e.source for e in events)),
            'data': [e.data for e in events]
        }

    async def _send_to_queue(
        self,
        data: Dict[str, Any],
        processor_id: str
    ):
        """Send processed data to queue"""
        if processor_id not in self.processors:
            self.processors[processor_id] = asyncio.Queue()
            
        await self.processors[processor_id].put(data)

class StreamingPipeline:
    def __init__(
        self,
        kafka_bootstrap_servers: str,
        kafka_topic: str,
        processor: DataProcessor
    ):
        self.kafka_bootstrap_servers = kafka_bootstrap_servers
        self.kafka_topic = kafka_topic
        self.processor = processor
        self.logger = logging.getLogger(__name__)

    async def consume_kafka(self) -> AsyncIterator[DataEvent]:
        """Consume events from Kafka"""
        consumer = aiokafka.AIOKafkaConsumer(
            self.kafka_topic,
            bootstrap_servers=self.kafka_bootstrap_servers,
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        
        await consumer.start()
        
        try:
            async for msg in consumer:
                yield DataEvent(
                    id=str(msg.offset),
                    timestamp=datetime.utcnow(),
                    data=msg.value,
                    source='kafka'
                )
        finally:
            await consumer.stop()

    async def process_forever(self):
        """Process streaming data indefinitely"""
        while True:
            try:
                stream = self.consume_kafka()
                await self.processor.process_stream(
                    stream,
                    processor_id='kafka_processor'
                )
            except Exception as e:
                self.logger.error(f"Error in streaming pipeline: {str(e)}")
                await asyncio.sleep(5)  # Backoff before retry

# Usage Example
async def main():
    processor = DataProcessor(batch_size=100)
    pipeline = StreamingPipeline(
        kafka_bootstrap_servers='localhost:9092',
        kafka_topic='data_stream',
        processor=processor
    )
    
    await pipeline.process_forever()

if __name__ == "__main__":
    asyncio.run(main())
```

[Content continues with sections on Advanced Debugging, Security Patterns, etc. - truncated for brevity]

Would you like me to continue with more sections on:
1. Advanced Debugging Techniques
2. Security Patterns and Best Practices
3. Data Validation and Schema Evolution
4. Advanced Testing Patterns
5. Architectural Patterns for Large-Scale Applications

Let me know which areas you'd like to explore next!
