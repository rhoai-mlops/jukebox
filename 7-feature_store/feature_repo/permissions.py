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


namespaces = ["<USER_NAME>-toolings"] # Update this 👈

all_resources = Permission(
    name="all_resources",
    types=ALL_RESOURCE_TYPES,
    policy=NamespaceBasedPolicy(namespaces=namespaces),
    actions=[AuthzedAction.DESCRIBE] + READ
)