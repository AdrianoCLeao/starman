use crate::camera::camera::Camera;
use crate::planar_camera::PlanarCamera;
use crate::post_processing::post_processing_effect::PostProcessingEffect;
use crate::renderer::renderer::Renderer;
use crate::window::window::Window;

pub trait State: 'static {
    fn step(&mut self, window: &mut Window);

    #[deprecated(
        note = "This will be replaced by `.cameras_and_effect_and_renderer` which is more flexible."
    )]
    fn cameras_and_effect(
        &mut self,
    ) -> (
        Option<&mut dyn Camera>,
        Option<&mut dyn PlanarCamera>,
        Option<&mut dyn PostProcessingEffect>,
    ) {
        (None, None, None)
    }

    fn cameras_and_effect_and_renderer(
        &mut self,
    ) -> (
        Option<&mut dyn Camera>,
        Option<&mut dyn PlanarCamera>,
        Option<&mut dyn Renderer>,
        Option<&mut dyn PostProcessingEffect>,
    ) {
        #[allow(deprecated)]
        let res = self.cameras_and_effect(); 
        (res.0, res.1, None, res.2)
    }
}

impl State for () {
    fn step(&mut self, _: &mut Window) {}
}
