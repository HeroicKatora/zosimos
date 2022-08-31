//! The editor state itself, sans causal snapshot system.
use crate::compute::Compute;
use crate::surface::Surface;
use crate::winit::{ModalContext, ModalEditor, ModalEvent, RedrawError, Window};

#[derive(Default)]
pub struct Editor {
    close_requested: bool,
    num_frames: usize,
}

impl ModalEditor for Editor {
    type Compute = Compute;
    type Surface = Surface;

    fn event(&mut self, ev: ModalEvent, _: &mut ModalContext) {
        log::info!("{:?}", ev);
    }

    fn redraw_request(&mut self, surface: &mut Surface) -> Result<(), RedrawError> {
        if let Err(e) = self.draw_to_surface(surface) {
            self.drawn_error(e)?;
        }

        self.num_frames += 1;
        Ok(())
    }

    fn exit(&self) -> bool {
        self.close_requested
    }

    fn lost(&mut self, surface: &mut Self::Surface) {
        if let Err(e) = surface.lost() {
            log::error!("Lost window , failed to reinitialized: {:?}", e);
            self.close_requested = true;
        }
    }

    fn outdated(&mut self, surface: &mut Self::Surface) {
        if let Err(e) = surface.outdated() {
            log::error!("Outdated window surface, failed to reinitialized: {:?}", e);
            self.close_requested = true;
        }
    }
}

impl Editor {
    pub fn draw_to_surface(&mut self, surface: &mut Surface) -> Result<(), wgpu::SurfaceError> {
        let start = std::time::Instant::now();
        let full_start = start;
        let mut texture = surface.get_current_texture()?;
        let end = std::time::Instant::now();
        log::warn!("Time get texture{:?}", end.saturating_duration_since(start));
        let start = end;
        surface.present_to_texture(&mut texture);
        let end = std::time::Instant::now();
        log::warn!(
            "Time rendering total {:?}",
            end.saturating_duration_since(start)
        );
        texture.present();
        let end = std::time::Instant::now();
        log::warn!(
            "Time present total {:?}",
            end.saturating_duration_since(full_start)
        );
        Ok(())
    }

    pub fn drawn_error(
        &mut self,
        err: wgpu::SurfaceError,
    ) -> Result<(), RedrawError> {
        Ok(match err {
            wgpu::SurfaceError::Lost => {
                return Err(RedrawError::Lost);
            }
            wgpu::SurfaceError::OutOfMemory => {
                log::warn!("Out-of-Memory, closing now");
                self.close_requested = true
            }
            wgpu::SurfaceError::Outdated => {
                return Err(RedrawError::Outdated);
            }
            e => log::warn!("{:?}", e),
        })
    }
}
