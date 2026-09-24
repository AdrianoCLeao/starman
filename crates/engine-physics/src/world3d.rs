use bevy_ecs::prelude::Resource;
use engine_core::DEFAULT_FIXED_TIMESTEP_SECONDS;
use rapier3d::prelude::*;
use std::sync::Mutex;

/// Collects Rapier collision events during a step.
#[derive(Default)]
pub struct PhysicsEventQueue {
    collisions: Mutex<Vec<CollisionEvent>>,
}

impl PhysicsEventQueue {
    /// Takes the events collected since the last drain.
    pub fn drain(&self) -> Vec<CollisionEvent> {
        std::mem::take(&mut *self.collisions.lock().unwrap_or_else(|p| p.into_inner()))
    }
}

impl EventHandler for PhysicsEventQueue {
    fn handle_collision_event(
        &self,
        _bodies: &RigidBodySet,
        _colliders: &ColliderSet,
        event: CollisionEvent,
        _contact_pair: Option<&ContactPair>,
    ) {
        self.collisions
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .push(event);
    }

    fn handle_contact_force_event(
        &self,
        _dt: Real,
        _bodies: &RigidBodySet,
        _colliders: &ColliderSet,
        _contact_pair: &ContactPair,
        _total_force_magnitude: Real,
    ) {
    }
}

#[derive(Resource)]
pub struct PhysicsWorld3D {
    pub rigid_body_set: RigidBodySet,
    pub collider_set: ColliderSet,
    pub gravity: Vector<Real>,
    pub integration_parameters: IntegrationParameters,
    pub physics_pipeline: PhysicsPipeline,
    pub island_manager: IslandManager,
    pub broad_phase: DefaultBroadPhase,
    pub narrow_phase: NarrowPhase,
    pub impulse_joint_set: ImpulseJointSet,
    pub multibody_joint_set: MultibodyJointSet,
    pub ccd_solver: CCDSolver,
    pub query_pipeline: QueryPipeline,
    pub events: PhysicsEventQueue,
    /// Fixed body at the origin that world-anchored joints attach to.
    world_anchor: Option<RigidBodyHandle>,
}

impl Default for PhysicsWorld3D {
    fn default() -> Self {
        Self::new()
    }
}

impl PhysicsWorld3D {
    pub fn new() -> Self {
        Self::with_timestep(DEFAULT_FIXED_TIMESTEP_SECONDS as f32)
    }

    pub fn with_timestep(fixed_dt_seconds: f32) -> Self {
        let integration_parameters = IntegrationParameters {
            dt: fixed_dt_seconds.max(0.000_001),
            ..IntegrationParameters::default()
        };

        Self {
            rigid_body_set: RigidBodySet::new(),
            collider_set: ColliderSet::new(),
            gravity: vector![0.0, -9.81, 0.0],
            integration_parameters,
            physics_pipeline: PhysicsPipeline::new(),
            island_manager: IslandManager::new(),
            broad_phase: DefaultBroadPhase::new(),
            narrow_phase: NarrowPhase::new(),
            impulse_joint_set: ImpulseJointSet::new(),
            multibody_joint_set: MultibodyJointSet::new(),
            ccd_solver: CCDSolver::new(),
            query_pipeline: QueryPipeline::new(),
            events: PhysicsEventQueue::default(),
            world_anchor: None,
        }
    }

    pub fn set_timestep(&mut self, fixed_dt_seconds: f32) {
        self.integration_parameters.dt = fixed_dt_seconds.max(0.000_001);
    }

    pub fn timestep_seconds(&self) -> f32 {
        self.integration_parameters.dt
    }

    pub fn step(&mut self) {
        self.physics_pipeline.step(
            &self.gravity,
            &self.integration_parameters,
            &mut self.island_manager,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.rigid_body_set,
            &mut self.collider_set,
            &mut self.impulse_joint_set,
            &mut self.multibody_joint_set,
            &mut self.ccd_solver,
            Some(&mut self.query_pipeline),
            &(),
            &self.events,
        );
    }

    /// Refreshes the query acceleration structure without stepping (after
    /// adding colliders, before the first step).
    pub fn update_query_pipeline(&mut self) {
        self.query_pipeline.update(&self.collider_set);
    }

    /// The shared fixed body joints attach to when they have no target.
    pub fn world_anchor(&mut self) -> RigidBodyHandle {
        if let Some(handle) = self.world_anchor {
            if self.rigid_body_set.contains(handle) {
                return handle;
            }
        }
        let handle = self
            .rigid_body_set
            .insert(RigidBodyBuilder::fixed().build());
        self.world_anchor = Some(handle);
        handle
    }
}

#[cfg(test)]
#[path = "world3d_tests.rs"]
mod tests;
