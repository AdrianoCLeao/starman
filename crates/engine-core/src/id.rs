//! Stable, persistable identifiers, per ADR 0002.
//!
//! Projects, source assets, and scene entities are identified by UUID v4.
//! Sub-assets are identified by UUID v5, derived deterministically from their
//! parent asset id plus a stable importer key. These ids are what gets
//! persisted to disk (as lowercase hyphenated strings) — the ECS `Entity`
//! value is a runtime handle only and is never persisted as identity.

use std::fmt;
use std::str::FromStr;

use bevy_ecs::component::Component;
use serde::{de::Error as DeError, Deserialize, Deserializer, Serialize, Serializer};
use uuid::Uuid;

use crate::error::{EngineError, Result};

macro_rules! define_persistent_id {
    ($name:ident, $doc:expr) => {
        #[doc = $doc]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub struct $name(Uuid);

        impl $name {
            /// Generates a new random (v4) id.
            pub fn new_v4() -> Self {
                Self(Uuid::new_v4())
            }

            /// Wraps an existing UUID as this id type.
            pub fn from_uuid(uuid: Uuid) -> Self {
                Self(uuid)
            }

            /// Returns the underlying UUID.
            pub fn as_uuid(self) -> Uuid {
                self.0
            }

            /// Parses a lowercase hyphenated UUID string into this id type.
            pub fn parse(value: &str) -> Result<Self> {
                Uuid::parse_str(value)
                    .map(Self)
                    .map_err(|error| EngineError::InvalidId {
                        kind: stringify!($name),
                        value: value.to_owned(),
                        reason: error.to_string(),
                    })
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}", self.0.hyphenated())
            }
        }

        impl FromStr for $name {
            type Err = EngineError;

            fn from_str(value: &str) -> Result<Self> {
                Self::parse(value)
            }
        }

        impl Serialize for $name {
            fn serialize<S: Serializer>(
                &self,
                serializer: S,
            ) -> std::result::Result<S::Ok, S::Error> {
                serializer.serialize_str(&self.to_string())
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D: Deserializer<'de>>(
                deserializer: D,
            ) -> std::result::Result<Self, D::Error> {
                let raw = String::deserialize(deserializer)?;
                Self::parse(&raw).map_err(DeError::custom)
            }
        }
    };
}

define_persistent_id!(
    ProjectId,
    "Stable identifier for a Starman project (UUID v4)."
);
define_persistent_id!(EntityId, "Stable identifier for a scene entity (UUID v4).");
define_persistent_id!(
    SourceAssetId,
    "Stable identifier for a source asset (UUID v4)."
);

impl EntityId {
    /// Deterministically derives a fallback id for an entity that does not
    /// carry a [`PersistentId`] component, from a stable discriminant (e.g. a
    /// traversal path within the scene being serialized). This exists only so
    /// that re-serializing an unchanged, never-loaded world is reproducible
    /// within a single process; entities that have round-tripped through a
    /// scene file always carry a real, stable id via [`PersistentId`].
    pub fn deterministic_fallback(discriminant: &str) -> Self {
        Self(Uuid::new_v5(&Uuid::NAMESPACE_OID, discriminant.as_bytes()))
    }
}

/// Stable identifier for a sub-asset (UUID v5), derived from its parent
/// [`SourceAssetId`] plus a stable importer key (e.g. "mesh:0" for the first
/// mesh extracted from a glTF file).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SubAssetId(Uuid);

impl SubAssetId {
    pub fn derive(parent: SourceAssetId, importer_key: &str) -> Self {
        Self(Uuid::new_v5(&parent.as_uuid(), importer_key.as_bytes()))
    }

    pub fn from_uuid(uuid: Uuid) -> Self {
        Self(uuid)
    }

    pub fn as_uuid(self) -> Uuid {
        self.0
    }

    pub fn parse(value: &str) -> Result<Self> {
        Uuid::parse_str(value)
            .map(Self)
            .map_err(|error| EngineError::InvalidId {
                kind: "SubAssetId",
                value: value.to_owned(),
                reason: error.to_string(),
            })
    }
}

impl fmt::Display for SubAssetId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0.hyphenated())
    }
}

impl FromStr for SubAssetId {
    type Err = EngineError;

    fn from_str(value: &str) -> Result<Self> {
        Self::parse(value)
    }
}

impl Serialize for SubAssetId {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for SubAssetId {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let raw = String::deserialize(deserializer)?;
        Self::parse(&raw).map_err(DeError::custom)
    }
}

/// Attaches a stable [`EntityId`] to a live ECS entity. Present on every
/// entity that has round-tripped through a scene file; entities spawned only
/// in memory (never loaded or saved) may not carry one yet.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub struct PersistentId(pub EntityId);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_through_display_and_parse() {
        let id = EntityId::new_v4();
        let parsed = EntityId::parse(&id.to_string()).expect("id should parse back");
        assert_eq!(id, parsed);
    }

    #[test]
    fn display_is_lowercase_hyphenated() {
        let id = ProjectId::from_uuid(Uuid::nil());
        assert_eq!(id.to_string(), "00000000-0000-0000-0000-000000000000");
    }

    #[test]
    fn parse_rejects_invalid_uuid() {
        let error = EntityId::parse("not-a-uuid").expect_err("should reject invalid uuid");
        assert!(error.to_string().contains("EntityId"));
    }

    #[test]
    fn sub_asset_id_is_deterministic_for_same_parent_and_key() {
        let parent = SourceAssetId::new_v4();
        let a = SubAssetId::derive(parent, "mesh:0");
        let b = SubAssetId::derive(parent, "mesh:0");
        assert_eq!(a, b);
    }

    #[test]
    fn sub_asset_id_differs_by_importer_key() {
        let parent = SourceAssetId::new_v4();
        let a = SubAssetId::derive(parent, "mesh:0");
        let b = SubAssetId::derive(parent, "mesh:1");
        assert_ne!(a, b);
    }

    #[test]
    fn entity_id_fallback_is_deterministic_for_same_discriminant() {
        let a = EntityId::deterministic_fallback("0/1");
        let b = EntityId::deterministic_fallback("0/1");
        assert_eq!(a, b);
    }

    #[test]
    fn serde_round_trip_preserves_value() {
        let id = EntityId::new_v4();
        let serialized = ron::to_string(&id).expect("id should serialize");
        let deserialized: EntityId = ron::from_str(&serialized).expect("id should deserialize");
        assert_eq!(id, deserialized);
    }
}
