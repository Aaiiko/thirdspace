import { BusinessExplore } from "@/app/modules/business/BusinessExplore";
import Toolbar from "@/app/modules/Toolbar";
import { Business } from "@/types/BusinessTypes";
import { Metadata } from "next";
export const metadata: Metadata = {
    title: 'For You',
    description: '...',
}

export default function forYouPage() {
    const MOCK_DATA: Business[] = [
        {
            name: "The Greek Taverna",
            description: "Authentic Greek cuisine with a modern twist.",
            review: 4.5,
            price: 3,
            location: "123 Main St, Springfield",
            category: "Greek",
            image: "https://images.squarespace-cdn.com/content/v1/5e484ab628c78d6f7e602d73/c828d00f-3184-44d0-9109-eea1b960075d/greek-foods-souvlaki.jpg",
            id: "1"
        },
        {
            name: "Pizza Paradise",
            description: "Delicious wood-fired pizzas with fresh ingredients.",
            review: 4.8,
            price: 2,
            location: "456 Elm St, Springfield",
            category: "Pizza",
            image: "https://th.bing.com/th/id/R.03193e7f4365fa8921502a3f1b85e1b6?rik=FXpqFP8IoYkIdw&riu=http%3a%2f%2f3.bp.blogspot.com%2f-rxO9iIBpdyo%2fUeUl4yaOLtI%2fAAAAAAAAFtk%2fk0_xm7L25Zo%2fs1600%2fPH2.jpg&ehk=nacC0xWk95PXDV1ATfNf%2fv1y53hbsouC9x8R%2bYW1DLs%3d&risl=&pid=ImgRaw&r=0",
            id: "2"
        },
        {
            name: "Sushi World",
            description: "A wide variety of sushi and sashimi, made to order.",
            review: 4.7,
            price: 4,
            location: "789 Oak St, Springfield",
            category: "Sushi",
            image: "https://th.bing.com/th/id/OIP.yS3PzX_bQcxiGYKUYRN2iwHaF7?rs=1&pid=ImgDetMain",
            id: "3"
        },
        {
            name: "Burger Haven",
            description: "Gourmet burgers with unique toppings and sides.",
            review: 4.3,
            price: 2,
            location: "101 Pine St, Springfield",
            category: "Burgers",
            image: "https://example.com/image7.jpg",
            id: "4"
        },
        {
            name: "Taco Fiesta",
            description: "A fiesta of flavors with our authentic Mexican tacos.",
            review: 4.6,
            price: 1,
            location: "202 Maple St, Springfield",
            category: "Mexican",
            image: "https://www.samtell.com/hs-fs/hubfs/Blogs/Four-Scrumptous-Tacos-Lined-up-with-ingredients-around-them-1.jpg?width=1800&name=Four-Scrumptous-Tacos-Lined-up-with-ingredients-around-them-1.jpg",
            id: "5"
        }
    ];

    return (
        <div className="bg-[#f4f4f4]">
            <div className="flex justify-center items-center min-h-screen">
                <BusinessExplore businesses={MOCK_DATA} />
            </div>
        </div>
    );
}