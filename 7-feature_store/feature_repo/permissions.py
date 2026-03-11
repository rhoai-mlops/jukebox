# Example permissions configuration with groups and namespaces support
# This demonstrates how to use the new group-based and namespace-based policies
# in addition to the existing role-based policies

from feast.feast_object import ALL_FEATURE_VIEW_TYPES, ALL_RESOURCE_TYPES
from feast.project import Project
from feast.entity import Entity
from feast.feature_view import FeatureView
from feast.on_demand_feature_view import OnDemandFeatureView
from feast.batch_feature_view import BatchFeatureView
from feast.stream_feature_view import StreamFeatureView
from feast.feature_service import FeatureService
from feast.data_source import DataSource
from feast.saved_dataset import SavedDataset
from feast.permissions.permission import Permission
from feast.permissions.action import READ, AuthzedAction, ALL_ACTIONS
from feast.permissions.policy import RoleBasedPolicy, GroupBasedPolicy, NamespaceBasedPolicy, CombinedGroupNamespacePolicy
import os

user_name = os.environ.get("AWS_ACCESS_KEY_ID", "default")
namespaces = [f"{user_name}-toolings", f"{user_name}-jukebox"]

all_resources_read = Permission(
    name="all_resources_read",
    types=ALL_RESOURCE_TYPES,
    policy=RoleBasedPolicy(roles=["feast-reader", "feast-writer"]),
    actions=[AuthzedAction.DESCRIBE] + READ
)

all_resources_write = Permission(
    name="all_resources_write",
    types=ALL_RESOURCE_TYPES,
    policy=RoleBasedPolicy(roles=["feast-writer"]),
    actions=ALL_ACTIONS
)

all_resources_ns = Permission(
    name="all_resources_ns",
    types=ALL_RESOURCE_TYPES,
    policy=NamespaceBasedPolicy(namespaces=namespaces),
    actions=ALL_ACTIONS
)
