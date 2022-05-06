use std::f64::consts::PI;

use image::{error::ImageError, ImageBuffer, Rgb};
use rand::{distributions::Distribution, distributions::Uniform, thread_rng, Rng};

#[derive(Clone, Copy, Debug)]
pub(crate) struct Vec3 {
    x: f64,
    y: f64,
    z: f64,
}

impl Vec3 {
    fn new<T: Into<f64>>(x: T, y: T, z: T) -> Self {
        Vec3 {
            x: x.into(),
            y: y.into(),
            z: z.into(),
        }
    }

    fn length(&self) -> f64 {
        (self.x.powf(2.0) + self.y.powf(2.0) + self.z.powf(2.0)).sqrt()
    }
    #[allow(dead_code)]
    fn normalize(&mut self) {
        let k = 1.0 / (self.x.powf(2.0) + self.y.powf(2.0) + self.z.powf(2.0)).sqrt();
        self.x *= k;
        self.y *= k;
        self.z *= k;
    }

    fn as_unit_vec(&self) -> Self {
        let len = self.length();
        if len == 0.0 {
            panic!("Zero length vec");
        }
        Self {
            x: self.x / len,
            y: self.y / len,
            z: self.z / len,
        }
    }

    fn random_in_unit_sphere() -> Self {
        let mut rng = thread_rng();
        loop {
            let v = Vec3::new(
                rng.gen_range(0.0..1.0),
                rng.gen_range(0.0..1.0),
                rng.gen_range(0.0..1.0),
            );
            if v.x * v.x + v.y * v.y + v.z + v.z > 1.0 {
                return v;
            }
        }
    }

    fn distance(&self, other: &Vec3) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2) + (self.z - other.z).powi(2))
            .sqrt()
    }

    fn dot(lhs: Self, rhs: Self) -> f64 {
        lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z
    }

    fn cross(lhs: Self, rhs: Self) -> Self {
        Vec3 {
            x: lhs.y * rhs.z - lhs.z * rhs.y,
            y: -(lhs.x * rhs.z - lhs.z * rhs.x),
            z: lhs.x * rhs.y - lhs.y * rhs.x,
        }
    }
}

impl PartialEq for Vec3 {
    fn eq(&self, other: &Self) -> bool {
        if self.x.is_nan() || self.y.is_nan() || self.z.is_nan() {
            return false;
        }
        return self.x.eq(&other.x);
    }
}
impl std::ops::Add for Vec3 {
    type Output = Vec3;

    fn add(self, rhs: Self) -> Self::Output {
        Vec3 {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Vec3;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::Output {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}
impl std::ops::Neg for Vec3 {
    type Output = Self;
    fn neg(self) -> <Self as std::ops::Neg>::Output {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

impl std::ops::Mul<Vec3> for f64 {
    type Output = Vec3;

    fn mul(self, rhs: Vec3) -> Self::Output {
        Self::Output {
            x: self * rhs.x,
            y: self * rhs.y,
            z: self * rhs.z,
        }
    }
}

impl std::ops::Mul<f64> for Vec3 {
    type Output = Vec3;

    fn mul(self, rhs: f64) -> Self::Output {
        Self::Output {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}
impl std::ops::Mul for Vec3 {
    type Output = Self;

    fn mul(self, rhs: Vec3) -> Self::Output {
        Self::Output {
            x: self.x * rhs.x,
            y: self.y * rhs.y,
            z: self.z * rhs.z,
        }
    }
}
impl std::ops::Div<f64> for Vec3 {
    type Output = Vec3;

    fn div(self, rhs: f64) -> Self::Output {
        Vec3::new(self.x / rhs, self.y / rhs, self.z / rhs)
    }
}

impl std::ops::AddAssign for Vec3 {
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

#[allow(dead_code)]
enum Color {
    White,
    Red,
    Green,
    Blue,
    Black,
}

impl From<Color> for Vec3 {
    fn from(src: Color) -> Self {
        match src {
            Color::White => Vec3::new(1, 1, 1),
            Color::Red => Vec3::new(1, 0, 0),
            Color::Green => Vec3::new(0, 1, 0),
            Color::Blue => Vec3::new(0, 0, 1),
            Color::Black => Vec3::new(0, 0, 0),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Ray {
    origin: Vec3,
    direction: Vec3,
}

impl Ray {
    fn new(origin: Vec3, direction: Vec3) -> Self {
        Self { origin, direction }
    }

    fn point_at_parameter(&self, t: f64) -> Vec3 {
        self.origin + (t * self.direction)
    }
}

struct Hit<'a> {
    t: f64,
    p: Vec3,
    normal: Vec3,
    material: &'a Material,
}

impl<'a> Hit<'a> {
    fn new(t: f64, p: Vec3, normal: Vec3, material: &'a Material) -> Self {
        Hit {
            t,
            p,
            normal,
            material,
        }
    }
}

trait Hittable {
    fn hit(&self, r: &Ray, t_min: f64, t_max: f64) -> Option<Hit>;
}

struct World {
    objects: Vec<Box<dyn Hittable + Sync>>,
}

impl World {
    fn new() -> Self {
        Self { objects: vec![] }
    }

    fn build(mut self, o: Box<dyn Hittable + Sync>) -> Self {
        self.objects.push(o);
        self
    }

    fn add(&mut self, o: Box<dyn Hittable + Sync + Send>) {
        self.objects.push(o);
    }
}

impl Hittable for World {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<Hit> {
        let mut closest = t_max;
        let mut winner: Option<Hit> = None;

        for o in &self.objects {
            match o.hit(ray, t_min, closest) {
                Some(h) => {
                    closest = h.t;
                    winner = Some(h);
                }
                None => (),
            }
        }
        return winner;
    }
}

#[allow(dead_code)]
enum Material {
    Simple(Vec3),
    Lambertian(Vec3),
    Metal(Vec3, f64),
    Dielectric(Vec3, f64),
}

impl Material {
    fn color(&self, world: &World, ray: &Ray, hit: &Hit, depth: u32) -> Vec3 {
        fn reflect(v: Vec3, n: Vec3) -> Vec3 {
            v - (2.0 * Vec3::dot(v, n) * n)
        }

        fn refract(v: Vec3, n: Vec3, ni_over_nt: f64) -> Option<Vec3> {
            let uv = v.as_unit_vec();
            let dt = Vec3::dot(uv, n);
            let discriminant = 1.0 - ni_over_nt * ni_over_nt * (1.0 - dt * dt);
            if discriminant > 0.0 {
                return Some(ni_over_nt * (uv - (n * dt)) - n * discriminant.sqrt());
            } else {
                return None;
            }
        }

        fn schlick(cos: f64, index: f64) -> f64 {
            let mut r0 = (1.0 - index) / (1.0 + index);
            r0 = r0 * r0;
            return r0 + (1.0 - r0) * (1.0 - cos).powf(5.0);
        }

        match self {
            Material::Simple(color) => {
                return color.clone();
            }
            Material::Lambertian(albedo) => {
                let target = hit.p + hit.normal + Vec3::random_in_unit_sphere();
                let scattered = Ray::new(hit.p, target - hit.p);
                return *albedo * color_at(&scattered, world, depth + 1);
            }
            Material::Metal(albedo, fuzz) => {
                let reflected = reflect(ray.direction.as_unit_vec(), hit.normal);
                let scattered =
                    Ray::new(hit.p, reflected + (*fuzz * Vec3::random_in_unit_sphere()));
                if Vec3::dot(scattered.direction, hit.normal) > 0.0 {
                    return *albedo * color_at(&scattered, world, depth + 1);
                } else {
                    // why do this?
                    return Color::Black.into();
                }
            }
            Material::Dielectric(albedo, index) => {
                let reflected = reflect(ray.direction, hit.normal);
                let outward_normal: Vec3;
                let ni_over_nt: f64;
                let cosine: f64;

                if Vec3::dot(ray.direction, hit.normal) > 0.0 {
                    outward_normal = -hit.normal;
                    ni_over_nt = *index;
                    cosine = index * Vec3::dot(ray.direction, hit.normal) / ray.direction.length();
                } else {
                    outward_normal = hit.normal;
                    ni_over_nt = 1.0 / index;
                    cosine = -Vec3::dot(ray.direction, hit.normal) / ray.direction.length();
                }

                if let Some(refracted) = refract(ray.direction, outward_normal, ni_over_nt) {
                    let reflect_prob = schlick(cosine, *index);
                    let mut rng = thread_rng();
                    if rng.gen_range(0.0..1.0) < reflect_prob {
                        return *albedo * color_at(&Ray::new(hit.p, reflected), world, depth + 1);
                    } else {
                        return *albedo * color_at(&Ray::new(hit.p, refracted), world, depth + 1);
                    }
                } else {
                    return color_at(&Ray::new(hit.p, reflected), world, depth + 1);
                }
            }
        }
    }
}

struct Point {
    p: Vec3,
    eps: f64,
    material: Material,
}

impl Point {
    #[allow(dead_code)]
    fn new(p: Vec3, eps: f64) -> Self {
        Self {
            p,
            eps,
            material: Material::Simple(Color::Red.into()),
        }
    }
}

impl Hittable for Point {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<Hit> {
        let d = &ray.direction;
        let r_sq = d.x * d.x + d.y * d.y + d.z * d.z;

        let t = (d.x * (self.p.x - ray.origin.x)
            + d.y * (self.p.y - ray.origin.y)
            + d.z * (self.p.z - ray.origin.z))
            / r_sq;

        if t > t_min && t < t_max {
            let closest = ray.origin + t * ray.direction;
            if self.p.distance(&closest) < self.eps {
                println!("hit at {:?}", closest);
                return Some(Hit::new(t, self.p.clone(), -ray.direction, &self.material));
            } else {
                return None;
            }
        } else {
            return None;
        }
    }
}
struct Sphere {
    center: Vec3,
    radius: f64,
    material: Material,
}

impl Sphere {
    fn new(center: Vec3, radius: f64, material: Material) -> Self {
        Self {
            center,
            radius,
            material,
        }
    }
}

impl Hittable for Sphere {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<Hit> {
        let oc = ray.origin - self.center;
        let a = Vec3::dot(ray.direction, ray.direction);
        let b = Vec3::dot(oc, ray.direction);
        let c = Vec3::dot(oc, oc) - self.radius * self.radius;
        let discriminant = b * b - a * c;
        if discriminant < 0.0 {
            None
        } else {
            let p = (-b - (b * b - a * c).sqrt()) / a;
            if p < t_max && p > t_min {
                let pp = ray.point_at_parameter(p);
                return Some(Hit::new(
                    p,
                    pp,
                    (pp - self.center) / self.radius,
                    &self.material,
                ));
            }
            let p = (-b + (b * b - a * c).sqrt()) / a;
            if p < t_max && p > t_min {
                let pp = ray.point_at_parameter(p);
                return Some(Hit::new(
                    p,
                    pp,
                    (pp - self.center) / self.radius,
                    &self.material,
                ));
            }
            return None;
        }
    }
}

#[derive(Debug)]
struct Camera {
    lower_left_corner: Vec3,
    u: Vec3,
    v: Vec3,
    #[allow(dead_code)]
    w: Vec3,
    horizontal: Vec3,
    vertical: Vec3,
    origin: Vec3,
    lens_radius: f64,
}

/*
impl Default for Camera {
    fn default() -> Self {
        Self {
            lower_left_corner: Vec3::new(-2, -1, -1),
            horizontal: Vec3::new(4, 0, 0),
            vertical: Vec3::new(0, 2, 0),
            origin: Vec3::new(0, 0, 0),
        }
    }
}
*/
impl Camera {
    fn new(lookfrom: Vec3, lookat: Vec3, vup: Vec3, vfov: f64, aspect: f64, aperture: f64) -> Self {
        let vup = vup.as_unit_vec();
        let theta = vfov * PI / 180.0;
        let half_height = (theta / 2.0).tan();
        let half_width = aspect * half_height;
        let w = (lookfrom - lookat).as_unit_vec();
        let u = Vec3::cross(vup, w).as_unit_vec();
        let v = Vec3::cross(w, u);
        let focal_len = (lookfrom - lookat).length();

        Self {
            lower_left_corner: lookfrom
                - (half_width * focal_len * u)
                - (half_height * focal_len * v)
                - focal_len * w,
            u,
            v,
            w,
            horizontal: 2.0 * focal_len * half_width * u,
            vertical: 2.0 * focal_len * half_height * v,
            origin: lookfrom,
            lens_radius: aperture / 2.0,
        }
    }

    fn get_ray(&self, s: f64, t: f64) -> Ray {
        let rd = self.lens_radius * Vec3::random_in_unit_sphere();
        let offset = self.u * rd.x + self.v * rd.y; // FIXME -- where's "z"?
        Ray::new(
            self.origin + offset,
            self.lower_left_corner + s * self.horizontal + t * self.vertical - self.origin - offset,
        )
    }
}

fn color_at(r: &Ray, world: &World, depth: u32) -> Vec3 {
    if let Some(hit) = world.hit(r, 0.001, f64::MAX) {
        if depth < 50 {
            return hit.material.color(world, r, &hit, depth + 1);
        } else {
            return Vec3::new(0, 0, 0);
        }
    } else {
        let ray_unit = r.direction.as_unit_vec();
        let t = 0.5 * (ray_unit.y + 1.0);
        return (1.0 - t) * Vec3::new(1.0, 1.0, 1.0) + t * Vec3::new(0.5, 0.7, 1.0);
    }
}

fn sample_color_at(
    camera: &Camera,
    world: &World,
    samples: u8,
    width: u32,
    height: u32,
    x: u32,
    y: u32,
) -> Vec3 {
    let mut rng = thread_rng();
    let mut colors = Vec3::new(0.0, 0.0, 0.0);
    for _ in 0..samples {
        let u = (x as f64 + rng.gen_range(0.0..0.3)) / width as f64;
        let v = ((height - y) as f64 + rng.gen_range(0.0..0.3)) / height as f64; // adjust for image coordinate system

        //let u = x as f64 / width as f64;
        //let v = (height - y) as f64 / height as f64;
        let r = camera.get_ray(u, v);
        //println!("ray to (u,v): ({}, {})", u, v);
        //println!("{:?}", r);
        colors += color_at(&r, &world, 0);
    }
    let colors = colors / samples as f64;
    return Vec3::new(colors.x.sqrt(), colors.y.sqrt(), colors.z.sqrt());
}
fn main() -> Result<(), ImageError> {
    // test commit to rayon branch
    let (width, height) = (1200, 900);
    let samples = 50;
    let mut world = World::new();
    world.add(Box::new(Sphere::new(
        Vec3::new(0, -1000, 0),
        1000.0,
        Material::Lambertian(Vec3::new(0.5, 0.5, 0.5)),
    )));
    let mut rng = thread_rng();
    let d = Uniform::new(0.0f64, 1.0f64);
    for a in -11..11 {
        for b in -11..11 {
            let m: f64 = rng.gen();
            let center = Vec3::new(
                a as f64 + 0.9 * d.sample(&mut rng),
                0.2,
                b as f64 + 0.9 * d.sample(&mut rng),
            );
            if (center - Vec3::new(4.0, 0.2, 0.0)).length() > 0.9 {
                world.add(Box::new(Sphere::new(
                    center,
                    0.2,
                    if m < 0.8 {
                        Material::Lambertian(Vec3::new(
                            d.sample(&mut rng) * d.sample(&mut rng),
                            d.sample(&mut rng) * d.sample(&mut rng),
                            d.sample(&mut rng) * d.sample(&mut rng),
                        ))
                    } else if m < 0.95 {
                        Material::Metal(
                            Vec3::new(
                                0.5 * (1.0 + d.sample(&mut rng)),
                                0.5 * (1.0 + d.sample(&mut rng)),
                                0.5 * (1.0 + d.sample(&mut rng)),
                            ),
                            0.0,
                        )
                    } else {
                        Material::Dielectric(Vec3::new(1, 1, 1), 1.10)
                    },
                )));
            }
        }
    }
    world.add(Box::new(Sphere::new(
        Vec3::new(0, 1, 0),
        1.0,
        Material::Dielectric(Vec3::new(1, 1, 1), 1.5),
    )));
    world.add(Box::new(Sphere::new(
        Vec3::new(-4, 1, 0),
        1.0,
        Material::Lambertian(Vec3::new(0.4, 0.2, 0.1)),
    )));
    world.add(Box::new(Sphere::new(
        Vec3::new(4, 1, 0),
        1.0,
        Material::Metal(Vec3::new(0.7, 0.6, 0.5), 0.1),
    )));

    let camera: Camera = Camera::new(
        Vec3::new(12.0, 2.0, 1.5),
        Vec3::new(0.0, 0.0, -1.0),
        Vec3::new(0, 1, 0),
        30.0,
        width as f64 / height as f64,
        0.1,
    );

    let mut img = ImageBuffer::new(width, height);

    rayon::scope(|s| {
        for (x, y, p) in img.enumerate_pixels_mut() {
            /*
             * shadow these variables to avoid capturing outer scope
             */
            let camera = &camera;
            let world = &world;
            s.spawn(move |_| {
                let colors = sample_color_at(camera, world, samples, width, height, x, y);

                *p = Rgb([
                    (255.0 * colors.x) as u8,
                    (255.0 * colors.y) as u8,
                    (255.0 * colors.z) as u8,
                ]);
            });
        }
    });
    img.save("file.jpg")?;
    Ok(())
}

#[cfg(test)]
mod test {

    use super::*;

    #[test]
    fn test_len() {
        assert_eq!(1.0, Vec3::new(0, 1, 0).length());
        assert_eq!(3.0, Vec3::new(3, 0, 0).length());
        assert_eq!(5.0, Vec3::new(3, 4, 0).length());
    }

    #[test]
    fn test_add_vec() {
        assert_eq!(Vec3::new(1, 1, 0), Vec3::new(1, 0, 0) + Vec3::new(0, 1, 0));
    }
}
