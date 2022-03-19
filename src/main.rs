use image::{error::ImageError, ImageBuffer, Rgb, RgbImage};
use rand::{thread_rng, Rng};

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
    fn normalize(&mut self) {
        let k = 1.0 / (self.x.powf(2.0) + self.y.powf(2.0) + self.z.powf(2.0)).sqrt();
        self.x *= k;
        self.y *= k;
        self.z *= k;
    }

    fn as_unit_vec(&self) -> Self {
        let len = self.length();
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

    fn dot(lhs: &Self, rhs: &Self) -> f64 {
        lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z
    }

    fn cross(lhs: &Self, rhs: &Self) -> Self {
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
impl std::ops::Add for &Vec3 {
    type Output = Vec3;

    fn add(self, rhs: Self) -> Self::Output {
        Vec3 {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl std::ops::Sub for &Vec3 {
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

impl std::ops::Mul<&Vec3> for f64 {
    type Output = Vec3;

    fn mul(self, rhs: &Vec3) -> Self::Output {
        Self::Output {
            x: self * rhs.x,
            y: self * rhs.y,
            z: self * rhs.z,
        }
    }
}

impl std::ops::Mul<f64> for &Vec3 {
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
            White => Vec3::new(1, 1, 1),
            Red => Vec3::new(1, 0, 0),
            Green => Vec3::new(0, 1, 0),
            Blue => Vec3::new(0, 0, 1),
            Black => Vec3::new(0, 0, 0),
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
        &self.origin + &(t * &self.direction)
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
    objects: Vec<Box<dyn Hittable>>,
}

impl World {
    fn new() -> Self {
        Self { objects: vec![] }
    }

    fn add(mut self, o: Box<dyn Hittable>) -> Self {
        self.objects.push(o);
        self
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

struct ScatterRecord {
    attenuation: Vec3,
    scattered: Ray,
}
impl ScatterRecord {
    fn new(attenuation: Vec3, scattered: Ray) -> Self {
        Self {
            attenuation,
            scattered,
        }
    }
}
trait Scatter {
    fn scatter(&self, ray: &Ray, hit: &Hit) -> Option<ScatterRecord>;
}

enum Material {
    Lambertian(Vec3),
    Metal(Vec3, f64),
    Dielectric(f64),
}

impl Scatter for Material {
    fn scatter(&self, ray: &Ray, hit: &Hit) -> Option<ScatterRecord> {
        fn reflect(v: &Vec3, n: &Vec3) -> Vec3 {
            v - &(2.0 * Vec3::dot(v, n) * n)
        }

        fn refract(v: &Vec3, n: &Vec3, ni_over_nt: f64) -> Option<Vec3> {
            let uv = v.as_unit_vec();
            let dt = Vec3::dot(&uv, &n);
            let discriminant = 1.0 - ni_over_nt * ni_over_nt * (1.0 - dt * dt);
            if discriminant > 0.0 {
                return Some(&(ni_over_nt * &(&uv - &(n * dt))) - &(n * discriminant.sqrt()));
            } else {
                return None;
            }
        }

        fn schlick(cos: f64, index: f64) -> f64 {
            let mut r0 = (1.0 - index) / (1.0 + index);
            r0 = r0 * r0;
            return r0 + (1.0 - r0) * (1.0 - cos).powf(5.0);
        }

        match (self) {
            Material::Lambertian(albedo) => {
                let target = &(hit.p) + &(&hit.normal + &Vec3::random_in_unit_sphere());
                let scattered = Ray::new(hit.p, &target - &hit.p);
                return Some(ScatterRecord::new(*albedo, scattered));
            }
            Material::Metal(albedo, fuzz) => {
                let reflected = reflect(&ray.direction.as_unit_vec(), &hit.normal);
                let scattered = Ray::new(
                    hit.p,
                    &reflected + &(*fuzz * &Vec3::random_in_unit_sphere()),
                );
                if Vec3::dot(&scattered.direction, &hit.normal) > 0.0 {
                    return Some(ScatterRecord::new(*albedo, scattered));
                } else {
                    return None;
                }
            }
            Material::Dielectric(index) => {
                let reflected = reflect(&ray.direction, &hit.normal);
                let outward_normal: Vec3;
                let ni_over_nt: f64;
                let cosine: f64;

                if Vec3::dot(&ray.direction, &hit.normal) > 0.0 {
                    outward_normal = -hit.normal;
                    ni_over_nt = *index;
                    cosine =
                        index * Vec3::dot(&ray.direction, &hit.normal) / ray.direction.length();
                } else {
                    outward_normal = hit.normal;
                    ni_over_nt = 1.0 / index;
                    cosine = -Vec3::dot(&ray.direction, &hit.normal) / ray.direction.length();
                }

                if let Some(refracted) = refract(&ray.direction, &outward_normal, ni_over_nt) {
                    let mut reflect_prob = schlick(cosine, *index);
                    let mut rng = thread_rng();
                    if rng.gen_range(0.0..1.0) < reflect_prob {
                        return Some(ScatterRecord::new(
                            Vec3::new(1, 1, 1),
                            Ray::new(hit.p, reflected),
                        ));
                    } else {
                        return Some(ScatterRecord::new(
                            Vec3::new(1, 1, 1),
                            Ray::new(hit.p, refracted),
                        ));
                    }
                } else {
                    return Some(ScatterRecord::new(
                        Vec3::new(1, 1, 1),
                        Ray::new(hit.p, reflected),
                    ));
                }
            }
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
        let oc = &ray.origin - &self.center;
        let a = Vec3::dot(&ray.direction, &ray.direction);
        let b = Vec3::dot(&oc, &ray.direction);
        let c = Vec3::dot(&oc, &oc) - self.radius * self.radius;
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
                    (&pp - &self.center) / self.radius,
                    &self.material,
                ));
            }
            let p = (-b + (b * b - a * c).sqrt()) / a;
            if p < t_max && p > t_min {
                let pp = ray.point_at_parameter(p);
                return Some(Hit::new(
                    p,
                    pp,
                    (&pp - &self.center) / self.radius,
                    &self.material,
                ));
            }
            return None;
        }
    }
}

struct Camera {
    lower_left_corner: Vec3,
    horizontal: Vec3,
    vertical: Vec3,
    origin: Vec3,
}

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
impl Camera {
    fn new(lower_left_corner: Vec3, horizontal: Vec3, vertical: Vec3, origin: Vec3) -> Self {
        Self {
            lower_left_corner,
            horizontal,
            vertical,
            origin,
        }
    }

    fn get_ray(&self, u: f64, v: f64) -> Ray {
        Ray::new(
            self.origin,
            &self.lower_left_corner + &(&(u * &self.horizontal) + &(v * &self.vertical)),
        )
    }
}

fn color(r: &Ray, world: &World, depth: u32) -> Vec3 {
    if let Some(hit) = world.hit(r, 0.001, f64::MAX) {
        if depth < 50 {
            if let Some(srec) = hit.material.scatter(r, &hit) {
                return srec.attenuation * color(&srec.scattered, world, depth + 1);
            } else {
                return Vec3::new(0, 0, 0);
            }
        } else {
            return Vec3::new(0, 0, 0);
        }
    } else {
        let ray_unit = r.direction.as_unit_vec();
        let t = 0.5 * (ray_unit.y + 1.0);
        let v2 = Vec3::new(0.5, 0.7, 1.0);
        return &((1.0 - t) * &Vec3::new(1.0, 1.0, 1.0)) + &(t * &Vec3::new(0.5, 0.7, 1.0));
    }
}

fn main() -> Result<(), ImageError> {
    let (width, height) = (400, 200);
    let samples = 50;
    let world = World::new()
        .add(Box::new(Sphere::new(
            Vec3::new(0, 0, -1),
            0.5,
            Material::Lambertian(Vec3::new(0.1, 0.2, 0.5)),
        )))
        .add(Box::new(Sphere::new(
            Vec3::new(0.0, -100.5, -1.0),
            100.0,
            Material::Lambertian(Vec3::new(0.8, 0.8, 0.0)),
        )))
        .add(Box::new(Sphere::new(
            Vec3::new(1, 0, -1),
            0.5,
            Material::Metal(Vec3::new(0.8, 0.6, 0.2), 0.4),
        )))
        .add(Box::new(Sphere::new(
            Vec3::new(-1, 0, -1),
            0.5,
            Material::Dielectric(1.5),
        )))
        .add(Box::new(Sphere::new(
            Vec3::new(-1, 0, -1),
            -0.45,
            Material::Dielectric(1.5),
        )));
    let camera: Camera = Default::default();
    let mut rng = thread_rng();
    let img: RgbImage = ImageBuffer::from_fn(width, height, |x, y| {
        let mut colors = Vec3::new(0.0, 0.0, 0.0);
        for _ in 0..samples {
            let u = (x as f64 + rng.gen_range(0.0..1.0)) / width as f64;
            let v = ((height - y) as f64 + rng.gen_range(0.0..1.0)) / height as f64; // adjust for image coordinate system
            let r = camera.get_ray(u, v);
            colors += color(&r, &world, 0);
        }
        let colors = colors / samples as f64;
        let colors = Vec3::new(colors.x.sqrt(), colors.y.sqrt(), colors.z.sqrt());
        Rgb([
            (255.0 * colors.x) as u8,
            (255.0 * colors.y) as u8,
            (255.0 * colors.z) as u8,
        ])
    });
    img.save("file.jpg")?;
    Ok(())
}
// TLIXSK
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
        assert_eq!(
            Vec3::new(1, 1, 0),
            &Vec3::new(1, 0, 0) + &Vec3::new(0, 1, 0)
        );
    }
}
