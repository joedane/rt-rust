use image::{error::ImageError, GenericImage, GenericImageView, ImageBuffer, Rgb, RgbImage};

#[derive(Clone, Copy, Debug)]
struct At(f64);

#[derive(Clone, Copy, Debug)]
struct Vec3 {
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
impl std::ops::Add for Vec3 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Vec3 {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
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

impl std::ops::Mul<Vec3> for At {
    type Output = Vec3;

    fn mul(self, rhs: Vec3) -> Self::Output {
        Vec3 {
            x: self.0 * rhs.x,
            y: self.0 * rhs.y,
            z: self.0 * rhs.z,
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
impl From<f64> for At {
    fn from(f: f64) -> Self {
        At(f)
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

    fn point_at_parameter(&self, t: At) -> Vec3 {
        self.origin + t * self.direction
    }
}

fn hit_sphere(center: &Vec3, radius: f64, ray: &Ray) -> bool {
    println!("origin: {:?}, direction: {:?}", ray.origin, ray.direction);
    let oc = ray.origin - *center;
    let a = Vec3::dot(&ray.direction, &ray.direction);
    let b = 2.0 * Vec3::dot(&oc, &ray.direction);
    let c = Vec3::dot(&oc, &oc) - radius * radius;
    let discriminant = b * b - 4.0 * a * c;
    println!("discrim: {}", discriminant);
    discriminant > 0.0
}

fn color(r: &Ray) -> Vec3 {
    if hit_sphere(&Vec3::new(0, 0, -1), 0.5, r) {
        println!("HIT");
        return Vec3::new(1, 0, 0);
    }
    let ray_unit = r.direction.as_unit_vec();
    let t = 0.5 * (ray_unit.y + 1.0);
    let v1 = Vec3::new(1, 1, 1);
    let v2 = Vec3::new(0.0, 10.7, 1.0);
    return ((1.0 - t) * v1 + t * v2).as_unit_vec();
}

fn main() -> Result<(), ImageError> {
    let (width, height) = (200, 100);
    let dx: f64 = 1f64 / width as f64;
    let dy: f64 = 1f64 / height as f64;

    let lower_left_corner = Vec3::new(-2, -1, -1);
    let horizontal = Vec3::new(4, 0, 0);
    let vertical = Vec3::new(0, 2, 0);
    let origin = Vec3::new(0, 0, 0);
    let img: RgbImage = ImageBuffer::from_fn(width, height, |x, y| {
        println!("(x, y) = ({}, {})", x, y);
        let u = dx * x as f64;
        let v = dy * y as f64;
        let r = Ray::new(origin, lower_left_corner + u * horizontal + v * vertical);
        let c = color(&r);
        Rgb([
            (255.0 * c.x) as u8,
            (255.0 * c.y) as u8,
            (255.0 * c.z) as u8,
        ])
    });
    img.save("file.jpg")?;
    Ok(())
}

mod test {

    use super::*;

    #[test]
    fn test_hit_sphere() {
        let r = Ray::new(Vec3::new(0, 0, 0), Vec3::new(0, 0, -1));
        assert!(hit_sphere(&Vec3::new(0, 0, -1), 05., &r));
    }
}
