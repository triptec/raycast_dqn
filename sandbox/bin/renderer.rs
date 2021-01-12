use crate::input::Input;
use core::option::Option::Some;
use geo::{Line, LineString, Point};
use sandbox::ray::Ray;
use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::pixels::Color;
use sdl2::render::WindowCanvas;
use sdl2::{pixels, rect, EventPump, Sdl};

pub struct Renderer {
    pub canvas: Option<WindowCanvas>,
    pub sdl_context: Sdl,
    pub event_pump: EventPump,
    pub display_width: f64,
    pub display_height: f64,
    pub resize: f64,
    pub render: bool,
    pub initialized: bool,
}

impl Renderer {
    pub fn new(_scalex: f64, _scaley: f64) -> Renderer {
        let sdl_context = sdl2::init().unwrap();
        let event_pump = sdl_context.event_pump().unwrap();
        Renderer {
            canvas: None,
            sdl_context,
            event_pump,
            display_width: 1000.0,
            display_height: 1000.0,
            resize: 0.2,
            render: false,
            initialized: false,
        }
    }
    pub fn init(&mut self) {
        if self.initialized {
            return;
        }
        let video_subsystem = self.sdl_context.video().unwrap();

        let window = video_subsystem
            .window(
                "rust-sdl2 demo: Video",
                self.display_width as u32,
                self.display_height as u32,
            )
            .position_centered()
            .opengl()
            .build()
            .map_err(|e| e.to_string())
            .unwrap();

        let mut canvas = window
            .into_canvas()
            .build()
            .map_err(|e| e.to_string())
            .unwrap();
        canvas.set_draw_color(Color::RGB(0, 0, 0));
        canvas.clear();
        canvas.present();
        self.canvas = Some(canvas);
        self.initialized = true;
    }
    pub fn get_input(&mut self) -> Input {
        for event in self.event_pump.poll_iter() {
            match event {
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(Keycode::Q),
                    ..
                } => return Input::Quit,
                Event::KeyDown {
                    keycode: Some(Keycode::R),
                    ..
                } => return Input::ToggleRender,
                Event::KeyDown {
                    keycode: Some(Keycode::E),
                    ..
                } => return Input::ToggleEvaluate,
                _ => return Input::None,
            }
        }
        Input::None
    }

    pub fn clear(&mut self) {
        if self.canvas.is_none() {
            return;
        }
        self.canvas
            .as_mut()
            .unwrap()
            .set_draw_color(Color::RGB(0, 0, 0));
        self.canvas.as_mut().unwrap().clear();
    }

    pub fn render_rays<C: Clone + Into<pixels::Color>>(
        &mut self,
        rays: &Vec<Ray>,
        color: C,
        center: &Point<f64>,
    ) {
        if self.canvas.is_none() {
            return;
        }
        if !self.render {
            return;
        }
        for ray in rays {
            self.render_lines(
                &ray.line_string.lines().into_iter().collect(),
                color.clone(),
                center,
            );
        }
    }
    fn offset(&self, center: &Point<f64>) -> (f64, f64) {
        let ax = (center.x() * self.display_width) * self.resize - self.display_width / 2.0;
        let ay = (center.y() * self.display_height) * self.resize - self.display_height / 2.0;
        return (-ax, -ay);
    }
    pub fn render_points<C: Into<pixels::Color>>(
        &mut self,
        points: &Vec<Point<f64>>,
        color: C,
        center: &Point<f64>,
    ) {
        if self.canvas.is_none() {
            return;
        }
        if !self.render {
            return;
        }
        let (ax, ay) = self.offset(center);
        self.canvas.as_mut().unwrap().set_draw_color(color);
        for point in points {
            self.canvas
                .as_mut()
                .unwrap()
                .draw_rect(rect::Rect::new(
                    (ax + point.x() * self.display_width * self.resize) as i32 - 5,
                    (ay + point.y() * self.display_height * self.resize) as i32 - 5,
                    10,
                    10,
                ))
                .unwrap();
        }
    }

    pub fn render_lines<C: Into<pixels::Color>>(
        &mut self,
        lines: &Vec<Line<f64>>,
        color: C,
        center: &Point<f64>,
    ) {
        if self.canvas.is_none() {
            return;
        }
        if !self.render {
            return;
        }
        let (ax, ay) = self.offset(center);
        self.canvas.as_mut().unwrap().set_draw_color(color);
        for line in lines {
            self.canvas
                .as_mut()
                .unwrap()
                .draw_line(
                    rect::Point::new(
                        (ax + line.start.x * self.display_width * self.resize) as i32,
                        (ay + line.start.y * self.display_height * self.resize) as i32,
                    ),
                    rect::Point::new(
                        (ax + line.end.x * self.display_width * self.resize) as i32,
                        (ay + line.end.y * self.display_height * self.resize) as i32,
                    ),
                )
                .unwrap();
        }
    }

    pub fn render_line_strings<C: Into<pixels::Color>>(
        &mut self,
        lines: &Vec<&LineString<f64>>,
        color: C,
        center: &Point<f64>,
    ) {
        if self.canvas.is_none() {
            return;
        }
        if !self.render {
            return;
        }
        let (ax, ay) = self.offset(center);
        self.canvas.as_mut().unwrap().set_draw_color(color);
        for line in lines {
            for line_segment in line.lines() {
                self.canvas
                    .as_mut()
                    .unwrap()
                    .draw_line(
                        rect::Point::new(
                            (ax + line_segment.start.x * self.display_width * self.resize) as i32,
                            (ay + line_segment.start.y * self.display_height * self.resize) as i32,
                        ),
                        rect::Point::new(
                            (ax + line_segment.end.x * self.display_width * self.resize) as i32,
                            (ay + line_segment.end.y * self.display_height * self.resize) as i32,
                        ),
                    )
                    .unwrap();
            }
        }
    }

    pub fn present(&mut self) {
        if self.canvas.is_none() {
            return;
        }
        self.canvas.as_mut().unwrap().present();
    }
}
